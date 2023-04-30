import os
import random
import subprocess

import numpy as np
from tqdm.auto import tqdm
from transformers import BertTokenizer, BertForMaskedLM, BertConfig

import torch
import torch.nn.functional as F

from model import TextDDPM, TextCNN

BLEU_SCRIPT_PATH = 'bleu/multi-bleu.perl'


class Evaluator:
    def __init__(self, args, trainer, dataloaders):
        self.args = args
        self.trainer = trainer

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.decoder = BertForMaskedLM(BertConfig())
        self.decoder.load_state_dict(torch.load(os.path.join(args.models_folder, 'bert_ae.pt')))
        self.decoder = self.decoder.cls.to(args.device)

        self.style_classifier = TextCNN(300, len(self.tokenizer)).to(args.device)
        self.style_classifier.load_state_dict(torch.load(os.path.join(args.models_folder, 'style_classifier.pt')))

        self.dataloaders = dataloaders

    def denormalize_output(self, x):
        return x * self.args.encoding_std + self.args.encoding_mean

    @torch.no_grad()
    def generate(self, ddpm: TextDDPM, batch, num_inference_steps=100, n_samples=10, seed=0):
        """
        :param batch: batch with encodings to extract context from
        :param decoder: linear decoder for text encodings
        :return:
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        device = next(ddpm.parameters()).device

        ddpm.eval()

        hidden_dim = batch['encodings'].shape[-1]
        latents = torch.randn(
            n_samples, self.args.max_text_len, hidden_dim,
            dtype=batch['encodings'].dtype, device=device
        )

        context = self.trainer.get_contex(batch)

        noise_scheduler = self.trainer.noise_scheduler
        noise_scheduler.set_timesteps(num_inference_steps, device=device)
        for i, t in enumerate(noise_scheduler.timesteps):
            with torch.no_grad():
                noise_pred = ddpm(
                    latents,
                    timesteps=t.unsqueeze(0).repeat(n_samples),
                    context=context
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = noise_scheduler.step(noise_pred, t, latents, eta=0.0).prev_sample

        with torch.no_grad():
            sample = self.decoder(self.denormalize_output(latents)).cpu()

        text = self.tokenizer.batch_decode(sample.argmax(-1))
        return text

    def get_loss(self, ddpm, dataloader):
        ddpm.eval()

        noise_scheduler = self.trainer.noise_scheduler
        num_timesteps = noise_scheduler.num_train_timesteps
        for batch in dataloader:
            latents = batch["encodings"].to(self.args.device)

            # add noise
            noise = torch.randn_like(latents)
            bs = latents.shape[0]
            timesteps = torch.randint(0, num_timesteps, (bs,), device=self.args.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            context = self.trainer.get_contex(batch)
            model_pred = ddpm(noisy_latents, timesteps, context=context)

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "sample":
                target = latents
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                )

            # DDPM LOSS
            rec_loss = (
                F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            )

            # TOTAL LOSS
            loss = rec_loss

            return loss.item()

    @staticmethod
    def bleu(reference_paths, model_predictions_path):

        """
        Given a file of hypothesis and reference files,
        evaluate the BLEU score using Moses scripts.
        """

        assert os.path.isfile(model_predictions_path)
        for ref in reference_paths:
            assert os.path.isfile(ref) or os.path.isfile(ref + '0')

        command = BLEU_SCRIPT_PATH + ' %s < %s'
        p = subprocess.Popen(command % (' '.join(reference_paths), model_predictions_path),
                             stdout=subprocess.PIPE, shell=True)
        result = p.communicate()[0].decode("utf-8")
        if result.startswith('BLEU'):
            return float(result[7:result.index(',')])
        else:
            print('Impossible to parse BLEU score! "%s"' % result)
            return -1

    def style_accuracy(self, predictions_path, style_label):
        """
        Given a file of hypothesis and target style,
        evaluate the Style Accuracy using fastText.
        """

        assert os.path.isfile(predictions_path)

        correct = []
        with open(predictions_path, 'r') as f:
            for line in f:
                tokenized = self.tokenizer(line, return_tensors='pt')['input_ids']
                with torch.inference_mode():
                    style_prob = self.style_classifier(tokenized.to(self.args.device)).cpu()
                correct.append(style_prob.round().item() == style_label)

        return np.mean(correct)

    def evaluate(self, ddpm: TextDDPM, style_label=None, predictions_path=None):
        results = {}
        metrics = self.args.evaluation_metrics
        if len(metrics) == 0:
            return results

        if style_label is None:
            dataloader = self.dataloaders[0]
        else:
            dataloader = self.dataloaders[style_label]

        if predictions_path is None:
            predictions_path = f"logs/{'%4x' % random.getrandbits(32)}.txt"
        source_path = f"logs/{'%4x' % random.getrandbits(32)}.txt"
        with open(predictions_path, 'w') as f1, open(source_path, 'w') as f2:
            for batch in tqdm(dataloader):
                if style_label is not None:
                    batch['labels'].fill_(1 - style_label)

                text = self.generate(ddpm, batch, n_samples=len(batch['text']))
                f1.write('\n'.join(text))
                f1.write('\n')
                f2.write('\n'.join(batch['text']))
                f2.write('\n')

        if 'bleu' in metrics:
            assert style_label is not None
            style_name = dataloader.dataset.idx_to_label[style_label]
            data_path = dataloader.dataset.data_paths[0]
            folder_path = '/'.join(data_path.split('/')[:-1])
            reference_paths = [os.path.join(folder_path, f'{style_name}.ref{i}') for i in range(4)]
            results['bleu'] = self.bleu(reference_paths, predictions_path)

        if 'self_bleu' in metrics:
            results['self_bleu'] = self.bleu([source_path], predictions_path)

        if 'style_accuracy' in metrics:
            results['style_accuracy'] = self.style_accuracy(predictions_path, 1 - style_label)

        results['val_loss'] = self.get_loss(ddpm, dataloader)

        return results
