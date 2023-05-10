import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tqdm.auto import tqdm, trange

from diffusers import DDIMScheduler, get_scheduler
from accelerate import Accelerator

import wandb
from transformers import BertForMaskedLM, BertConfig, BertTokenizerFast

from model import TextDDPM
from util import masked_std, masked_mean, dict_to_device, betas_for_alpha_bar

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Trainer:
    def __init__(self, args, ddpm: TextDDPM, context_tokenizer=None):
        self.ddpm = ddpm
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.context_tokenizer = context_tokenizer
        self.args = args

        print(self.ddpm.ddpm)
        self.ddmp_optimizer = torch.optim.AdamW(self.ddpm.ddpm.parameters(), lr=args.ddpm_lr)
        self.ddpm_lr_scheduler = get_scheduler(
            'cosine',
            optimizer=self.ddmp_optimizer,
            num_warmup_steps=args.num_warmup,
            num_training_steps=args.training_iters,
            num_cycles=1,
        )

        if args.train_context:
            self.context_optimizer = torch.optim.AdamW(self.ddpm.context_encoder.parameters(), lr=args.ce_lr)

        self.encoder = BertForMaskedLM(BertConfig())
        self.encoder.load_state_dict(torch.load(os.path.join(args.models_folder, 'bert_ae.pt')))
        self.encoder = self.encoder.eval()

        if args.noise_scheduler == 'sqrt':
            betas = betas_for_alpha_bar(
                args.num_train_timesteps,
                lambda t: 1 - np.sqrt(t + 0.0001),
            )
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=args.num_train_timesteps,
                trained_betas=betas,
                clip_sample=False,
                prediction_type="sample",
            )
        else:
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=args.num_train_timesteps,
                beta_start=0.001,
                beta_end=0.01,
                beta_schedule=args.noise_scheduler,
                trained_betas=None,
                clip_sample=False,
                prediction_type="sample"
            )

        # mean and std for encoding normalization
        self.mean = torch.load(f"data/encodings-bert_base-wiki-mean.pt")
        self.std = torch.load(f"data/encodings-bert_base-wiki-std.pt")

        self.accelerator = Accelerator(
            mixed_precision='fp16' if args.mp else 'no',
        )
        (
            self.ddpm, self.ddmp_optimizer, self.encoder,
            self.ddpm_lr_scheduler, self.mean, self.std, self.noise_scheduler
        ) = self.accelerator.prepare(
            self.ddpm, self.ddmp_optimizer, self.encoder,
            self.ddpm_lr_scheduler, self.mean, self.std, self.noise_scheduler
        )
        if args.train_context:
            self.context_optimizer = self.accelerator.prepare(self.context_optimizer)

        self.label_to_idx = {label: i for i, label in enumerate(args.labels)}
        self.idx_to_label = {i: label for i, label in enumerate(args.labels)}

        self.train_range = trange(1, self.args.training_iters + 1)
        self.train_range_iter = iter(self.train_range)
        self.step = 1

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean

    def encode_batch(self, batch):
        tokenized = self.tokenizer(
            batch["inputs"],
            max_length=self.args.max_text_len,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        tokenized['text'] = batch['inputs']
        if 'label' in batch:
            tokenized['labels'] = torch.tensor([self.label_to_idx[label] for label in batch['label']])

        device = self.accelerator.device
        with torch.no_grad():
            encodings = self.encoder.bert(
                input_ids=tokenized['input_ids'].to(device),
                attention_mask=tokenized['attention_mask'].to(device)
            )['last_hidden_state']
            encodings = self.normalize(encodings)

        tokenized['encodings'] = encodings

        return tokenized

    def get_contex(self, batch):
        """
        get the context for ddmp to condition on
        """
        if len(self.args.context_mode) == 0:
            return None

        if "context" in self.args.context_mode:
            context_inputs = self.context_tokenizer(
                batch['text'],
                padding=True,
                truncation=True,
                max_length=self.args.max_text_len,
                return_tensors='pt'
            )
            context_inputs = dict_to_device(context_inputs, self.accelerator.device)

            if self.args.train_context:
                context = self.ddpm.context_encoder(**context_inputs)['last_hidden_state']
            else:
                with torch.no_grad():
                    context = self.ddpm.context_encoder(**context_inputs)['last_hidden_state']

        if "style" in self.args.context_mode:
            style_markers = batch["labels"]
            # style_embeddings = self.get_sinusoidal_embedding(style_markers).unsqueeze(1)
            style_embeddings = style_markers[:, None, None].repeat(
                1, 1, self.args.context_dim
            ).to(self.accelerator.device).float()

        if self.args.context_mode == "style":
            print(style_embeddings.shape, flush=True)
            return style_embeddings
        elif self.args.context_mode == "context":
            return context
        else:
            return torch.cat([style_embeddings, context], dim=1)

    def train_one_epoch(self, dataloader) -> None:
        """
        :param ddmp: TextDiffusion model
        :param context_encoder: Model that encodes anything you need to condition on.
                                Trainable if `optimizer_context_encoder` is not None.
        :param dataloader: Train dataloader
        :param noise_scheduler:
        :param optimizer_ddpm:
        :param optimizer_context_encoder: None if `context_encoder` is frozen
        :param accelerator: Accelerator class for faster training
        :return: None
        """

        self.ddpm.train()
        device = self.accelerator.device

        # dataloader = self.accelerator.prepare(dataloader)

        num_timesteps = self.noise_scheduler.num_train_timesteps

        for batch in dataloader:
            _ = next(self.train_range_iter)
            self.step += 1

            batch = self.encode_batch(batch)
            x_0 = batch["encodings"]

            # add noise
            noise = torch.randn_like(x_0)
            bs = x_0.shape[0]
            timesteps = torch.randint(0, num_timesteps, (bs,), device=device).long()
            x_t = self.noise_scheduler.add_noise(x_0, noise, timesteps)

            context = self.get_contex(batch)
            attention_mask = batch['attention_mask'].to(device)
            model_pred = self.ddpm(x_t, timesteps, attention_mask=attention_mask, context=context)

            # Get the target for loss depending on the prediction type
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "sample":
                target = x_0
            else:
                raise ValueError(
                    f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
                )

            # DDPM LOSS
            model_pred, target = self.accelerator.gather_for_metrics((model_pred, target))
            rec_loss = F.mse_loss(model_pred, target, reduction="none").mean(-1)
            rec_loss *= attention_mask
            rec_loss = rec_loss.sum() / attention_mask.sum()

            # TOTAL LOSS
            loss = rec_loss

            self.accelerator.backward(loss)

            grad_norm = []
            for p in list(filter(lambda p: p.grad is not None, self.ddpm.ddpm.parameters())):
                grad_norm.append(p.grad.data.norm(2).item())
            grad_norm = np.mean(grad_norm)

            self.accelerator.clip_grad_norm_(self.ddpm.ddpm.parameters(), 1)
            self.ddmp_optimizer.step()
            self.ddpm_lr_scheduler.step()
            self.ddmp_optimizer.zero_grad()
            if self.args.train_context:
                self.context_optimizer.step()
                self.context_optimizer.zero_grad()

            decoder = self.encoder.cls
            with torch.no_grad():
                logits = decoder(self.denormalize(model_pred))
            bert_acc = self.bert_acc(batch['input_ids'].to(device), logits, attention_mask)

            wandb.log({
                'reconstruction_loss': rec_loss.item(),
                'grad_norm': grad_norm,
                'lr': self.ddpm_lr_scheduler.get_last_lr()[0],
                'x_t_mean': masked_mean(x_t, attention_mask).mean(),
                'x_t_std': masked_std(x_t, attention_mask).mean(),
                'x_0_mean': masked_mean(x_0, attention_mask).mean(),
                'x_0_std': masked_std(x_0, attention_mask).mean(),
                'bert_acc': bert_acc.item(),
            })

            if self.step % self.args.eval_every == 0:
                break

    @staticmethod
    def bert_acc(targets, logits, mask):
        pred_tokens = logits.argmax(dim=-1)

        mask = deepcopy(mask)
        mask.scatter_(dim=1, index=(mask.sum(dim=1) - 1).reshape(-1, 1), src=torch.zeros_like(mask))
        mask[:, 0] = 0
        return torch.sum(mask * (targets == pred_tokens)) / torch.sum(mask)
