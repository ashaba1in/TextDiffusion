import os
import random
import subprocess

import numpy as np
from tqdm.auto import tqdm
from transformers import BertTokenizerFast, GPT2LMHeadModel, GPT2Tokenizer

import torch
import torch.nn.functional as F

from model import TextDDPM, TextCNN
from util import dict_to_device

BLEU_SCRIPT_PATH = 'bleu/multi-bleu.perl'


class GPTMetric:
    def __init__(self, device="cpu"):
        self.name = "gpt2-large"
        self.model = GPT2LMHeadModel.from_pretrained(self.name).eval().to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.name)
        self.device = device

    @torch.no_grad()
    def __call__(self, text, reduce="mean"):
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = dict_to_device(inputs, self.device)

        loss = self.model(**inputs, labels=inputs["input_ids"]).loss.detach().cpu()
        num_tokens = torch.sum(inputs["attention_mask"]).item()
        if reduce == "sum":
            return loss.item() * num_tokens, num_tokens
        return loss.item(), num_tokens


class Evaluator:
    def __init__(self, args, trainer, dataloaders):
        self.args = args
        self.trainer = trainer

        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

        if 'style' in args.context_mode:
            self.style_classifier = TextCNN(300, len(self.tokenizer)).to(args.device)
            self.style_classifier.load_state_dict(torch.load(os.path.join(args.models_folder, 'style_classifier.pt')))

        self.bloom = GPTMetric(args.device)

        self.dataloaders = dataloaders

    @staticmethod
    def clear_text(texts):
        clear_texts = []
        for text in texts:
            # text = text.replace("[CLS]", "").replace("[PAD]", "").replace("[UNK]", "")
            # text = text.replace("[START]", "").replace("[END]", "")

            text = text.split()
            try:
                sep_pos = text.index('[SEP]')
            except:
                sep_pos = len(text)
            text = ' '.join(text[:sep_pos])

            clear_texts.append(text)

        return clear_texts

    @torch.no_grad()
    def generate(self, ddpm: TextDDPM, batch, num_inference_steps=200, n_samples=10, seed=0):
        """
        :param batch: batch with encodings to extract context from
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
                model_output = ddpm(
                    latents,
                    timesteps=t.unsqueeze(0).repeat(n_samples),
                    context=context
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = noise_scheduler.step(model_output, t, latents, eta=0.0).prev_sample
            # alpha_prod = noise_scheduler.alphas_cumprod[t]
            # next_t = noise_scheduler.timesteps[i + 1]
            # alpha_prod_prev = noise_scheduler.alphas_cumprod[next_t]
            # eps = (latents - torch.sqrt(alpha_prod) * model_output) / torch.sqrt(1 - alpha_prod)
            # latents = torch.sqrt(alpha_prod_prev) * model_output + torch.sqrt(1 - alpha_prod_prev) * eps

        denorm = self.trainer.denormalize
        decoder = self.trainer.encoder.cls
        with torch.no_grad():
            sample = decoder(denorm(latents)).cpu()

        text = self.tokenizer.batch_decode(sample.argmax(-1))

        return self.clear_text(text)

    def get_loss(self, ddpm, dataloader):
        ddpm.eval()

        noise_scheduler = self.trainer.noise_scheduler
        num_timesteps = noise_scheduler.num_train_timesteps
        for batch in dataloader:
            batch = self.trainer.encode_batch(batch)
            x_0 = batch["encodings"].to(self.args.device)

            # add noise
            noise = torch.randn_like(x_0)
            bs = x_0.shape[0]
            timesteps = torch.randint(0, num_timesteps, (bs,), device=self.args.device).long()
            x_t = noise_scheduler.add_noise(x_0, noise, timesteps)

            context = self.trainer.get_contex(batch)
            attention_mask = batch['attention_mask'].to(self.args.device)
            model_pred = ddpm(x_t, timesteps, attention_mask=attention_mask, context=context)

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "sample":
                target = x_0
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                )

            # DDPM LOSS
            rec_loss = F.mse_loss(model_pred.float(), target.float(), reduction="none").mean(-1)
            rec_loss *= attention_mask
            rec_loss = rec_loss.sum() / attention_mask.sum()

            # TOTAL LOSS
            loss = rec_loss

            return loss.item()

    def compute_bloom(self, texts):
        num_tokens = 0.0
        metric = 0.0
        for text in texts:
            t_metric, t_num = self.bloom(text, reduce="sum")
            if t_metric is None or np.isnan(t_metric):
                continue
            metric += t_metric
            num_tokens += t_num
        return metric / num_tokens

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

        if self.args.dataset == 'gyafc':
            correct = []
            with open(predictions_path, 'r') as f:
                for line in f:
                    tokenized = self.tokenizer(line, return_tensors='pt')['input_ids']
                    with torch.inference_mode():
                        style_prob = self.style_classifier(tokenized.to(self.args.device)).cpu()
                    correct.append(style_prob.round().item() == style_label)
            result = np.mean(correct)

        elif self.args.dataset == 'yelp':
            command = f'fastText/fasttext predict models/yelp_review_polarity.bin {predictions_path}'
            result = subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out = result.stdout.decode("utf-8")
            if len(out) == 0:
                print(result.stderr, flush=True)
                return 0

            labels = np.array([int(l[-1]) for l in out.strip().split('\n')])
            labels -= 1
            result = np.mean(labels == style_label)
        else:
            raise NotImplementedError('Unknown dataset')

        return result

    def evaluate(self, ddpm: TextDDPM, style_label=None, predictions_path=None):
        if style_label is None:
            dataloader = self.dataloaders[0]
        else:
            dataloader = self.dataloaders[style_label]

        if predictions_path is None:
            predictions_path = f"logs/{'%4x' % random.getrandbits(32)}.txt"
        source_path = f"logs/{'%4x' % random.getrandbits(32)}.txt"
        with open(predictions_path, 'w') as f1, open(source_path, 'w') as f2:
            for step, batch in tqdm(enumerate(dataloader)):
                batch = self.trainer.encode_batch(batch)
                if style_label is not None:
                    batch['labels'].fill_(1 - style_label)

                text = self.generate(ddpm, batch, n_samples=len(batch['text']))
                f1.write('\n'.join(text))
                f1.write('\n')
                f2.write('\n'.join(batch['text']))
                f2.write('\n')

                if step > 50:
                    break

        results = {}
        metrics = self.args.evaluation_metrics
        if 'self_bleu' in metrics:
            results['self_bleu'] = self.bleu([source_path], predictions_path)

        if 'bleu' in metrics and style_label is not None:
            style_name = dataloader.dataset.idx_to_label[style_label]
            data_path = dataloader.dataset.data_paths[0]
            folder_path = '/'.join(data_path.split('/')[:-1])
            reference_paths = [os.path.join(folder_path, f'{style_name}.ref{i}') for i in range(4)]
            results['bleu'] = self.bleu(reference_paths, predictions_path)

        if 'style_accuracy' in metrics and style_label is not None:
            results['style_accuracy'] = self.style_accuracy(predictions_path, 1 - style_label)

        if 'bloom' in metrics:
            with open(predictions_path, 'r') as f:
                texts = f.readlines()
            results['bloom'] = self.compute_bloom(texts)

        results['val_loss'] = self.get_loss(ddpm, dataloader)

        return results
