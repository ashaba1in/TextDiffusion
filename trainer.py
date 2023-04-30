import os

import numpy as np
import torch
import torch.nn.functional as F

from tqdm.auto import tqdm

from diffusers import DDIMScheduler, get_scheduler
from accelerate import Accelerator

import wandb

from model import TextDDPM

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Trainer:
    def __init__(self, args, ddpm: TextDDPM, context_tokenizer=None):
        self.ddpm = ddpm
        self.context_tokenizer = context_tokenizer
        self.args = args

        self.ddmp_optimizer = torch.optim.AdamW(self.ddpm.ddpm.parameters(), lr=args.ddpm_lr)
        self.ddpm_lr_scheduler = get_scheduler(
            'constant_with_warmup',
            optimizer=self.ddmp_optimizer,
            num_warmup_steps=1000,
            num_training_steps=1001,
        )
        if args.train_context:
            self.context_optimizer = torch.optim.AdamW(self.ddpm.context_encoder.parameters(), lr=args.ce_lr)

        if args.noise_scheduler == 'sqrt':
            betas = self.betas_for_alpha_bar(
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
                beta_end=0.02,
                beta_schedule=args.noise_scheduler,
                trained_betas=None,
                clip_sample=False,
                prediction_type="sample"
            )

        self.accelerator = Accelerator(
            mixed_precision='fp16',
        )
        self.ddpm, self.ddmp_optimizer = self.accelerator.prepare(
            self.ddpm, self.ddmp_optimizer
        )
        if args.train_context:
            self.context_optimizer = self.accelerator.prepare(self.context_optimizer)

    @staticmethod
    def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)

    def get_sinusoidal_embedding(self, x, dim):
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)

        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=x.device)
        emb = x.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    def get_contex(self, batch):
        """
        get the context for ddmp to condition on
        """

        if "context" in self.args.context_mode:
            context_inputs = self.context_tokenizer(
                batch['text'],
                padding=True,
                truncation=True,
                max_length=self.args.max_text_len,
                return_tensors='pt'
            )
            for key in context_inputs:
                context_inputs[key] = context_inputs[key].to(self.args.device)

            if self.args.train_context:
                context = self.ddpm.context_encoder(**context_inputs)['last_hidden_state']
            else:
                with torch.inference_mode():
                    context = self.ddpm.context_encoder(**context_inputs)['last_hidden_state']

        if "style" in self.args.context_mode:
            style_markers = batch["labels"].to(self.args.device)
            # style_embeddings = self.get_sinusoidal_embedding(style_markers).unsqueeze(1)
            style_embeddings = style_markers[:, None, None].repeat(1, 1, self.args.context_dim).float()

        if self.args.context_mode == "style":
            return style_markers
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

        dataloader = self.accelerator.prepare(dataloader)

        num_timesteps = self.noise_scheduler.num_train_timesteps

        for step, batch in enumerate(tqdm(dataloader)):
            latents = batch["encodings"].to(self.args.device)

            # add noise
            noise = torch.randn_like(latents)
            bs = latents.shape[0]
            timesteps = torch.randint(0, num_timesteps, (bs,), device=self.args.device).long()
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            context = self.get_contex(batch)
            model_pred = self.ddpm(noisy_latents, timesteps, context=context)

            # Get the target for loss depending on the prediction type
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "sample":
                target = latents
            else:
                raise ValueError(
                    f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
                )

            # DDPM LOSS
            rec_loss = (
                F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            )

            # TOTAL LOSS
            loss = rec_loss

            self.accelerator.backward(loss)

            grad_norm = []
            for p in list(filter(lambda p: p.grad is not None, self.ddpm.ddpm.parameters())):
                grad_norm.append(p.grad.data.norm(2).item())
            grad_norm = np.mean(grad_norm)

            self.ddmp_optimizer.step()
            self.ddpm_lr_scheduler.step()
            self.ddmp_optimizer.zero_grad()
            if self.args.train_context:
                self.context_optimizer.step()
                self.context_optimizer.zero_grad()

            wandb.log({
                'reconstruction_loss': rec_loss.item(),
                'grad_norm': grad_norm,
                'lr': self.ddpm_lr_scheduler.get_last_lr()[0],
            })
