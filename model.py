import argparse

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


class TextDDPM(nn.Module):
    def __init__(self, args: argparse.Namespace, ddpm, context_encoder=None):
        """
        :param ddpm: Transformer of your choice from hugging face
        :param context_encoder: Transformer of your choice from hugging face
        """
        super().__init__()
        self.ddpm = ddpm
        # ddpm_hidden_dim = args.ddpm_dim
        if not hasattr(self.ddpm, "input_projection"):
            if args.embedding_dim == args.ddpm_dim:
                self.ddpm.input_projection = nn.Identity()
                self.ddpm.output_projection = nn.Identity()
            else:
                self.ddpm.input_projection = nn.Sequential(
                    nn.Linear(args.embedding_dim, args.ddpm_dim),
                    nn.Tanh(),
                    nn.Linear(args.ddpm_dim, args.ddpm_dim)
                )
                self.ddpm.output_projection = nn.Sequential(
                    nn.Linear(args.ddpm_dim, args.ddpm_dim),
                    nn.Tanh(),
                    nn.Linear(args.ddpm_dim, args.embedding_dim)
                )
        if not hasattr(self.ddpm, "time_emb"):
            self.ddpm.time_emb = nn.Sequential(
                nn.Linear(args.hidden_t_dim, args.hidden_t_dim * 2),
                nn.SiLU(),
                nn.Linear(args.hidden_t_dim * 2, args.ddpm_dim),
            )
        if not hasattr(self.ddpm, "position_embeddings"):
            self.ddpm.position_embeddings = nn.Embedding(args.max_text_len, args.ddpm_dim)
        self.register_buffer("position_ids", torch.arange(args.max_text_len).expand((1, -1)))

        self.ddpm.layer_norm = nn.LayerNorm(args.ddpm_dim, eps=1e-12)
        self.ddpm.dropout = torch.nn.Dropout(0.1)

        self.context_encoder = context_encoder
        self.args = args

        assert self.args.context_mode in ["style", "context", "context+style", ""]

    @staticmethod
    def get_sinusoidal_embedding(x, dim):
        half_dim = dim // 2
        emb = np.log(10000) / half_dim

        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=x.device)
        emb = x.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    @staticmethod
    def get_extended_attention_mask(attention_mask, dtype):
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def forward(self, x_t, timesteps, attention_mask=None, context=None):
        if context is not None:
            context = context.to(x_t.dtype)
        seq_length = x_t.shape[1]
        emb_x = self.ddpm.input_projection(x_t)

        position_ids = self.position_ids[:, : seq_length]
        pos_emb = self.ddpm.position_embeddings(position_ids)

        timesteps_emb = self.get_sinusoidal_embedding(timesteps, self.args.hidden_t_dim)
        timesteps_emb = self.ddpm.time_emb(timesteps_emb)

        emb_inputs = emb_x + pos_emb + timesteps_emb[:, None, :]
        emb_inputs = self.ddpm.dropout(self.ddpm.layer_norm(emb_inputs))

        if attention_mask is not None:
            attention_mask = self.get_extended_attention_mask(attention_mask, emb_inputs.dtype)
        model_pred = self.ddpm(
            emb_inputs,
            encoder_hidden_states=context,
            attention_mask=attention_mask,
        )["last_hidden_state"]
        model_pred = self.ddpm.output_projection(model_pred)
        return model_pred

    def save(self, path):
        models_state_dict = {'ddpm': self.ddpm.state_dict()}
        if self.args.train_context:
            models_state_dict['context_encoder'] = self.context_encoder.state_dict()
        torch.save(models_state_dict, path)


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(EmbeddingLayer, self).__init__()
        self.embeding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        if len(x.size()) == 2:
            y = self.embeding(x)
        else:
            y = torch.matmul(x, self.embeding.weight)
        return y


class TextCNN(nn.Module):
    """A style classifier TextCNN"""

    def __init__(self, embed_dim, vocab_size, dropout=0.0):
        super().__init__()

        filter_sizes = [1, 2, 3, 4, 5]
        num_filters = [128, 128, 128, 128, 128]
        self.feature_dim = sum(num_filters)
        self.embeder = EmbeddingLayer(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, embed_dim))
            for (n, f) in zip(num_filters, filter_sizes)
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            self.dropout,
            nn.Linear(self.feature_dim, int(self.feature_dim / 2)), nn.ReLU(),
            nn.Linear(int(self.feature_dim / 2), 1)
        )

    def forward(self, inp):
        inp = self.embeder(inp).unsqueeze(1)
        if inp.shape[1] < 5:
            pad = inp.new_zeros(inp.shape[0], 5 - inp.shape[-2], inp.shape[-1])
            inp = torch.cat((inp, pad), -2)

        convs = [F.relu(conv(inp)).squeeze(3) for conv in self.convs]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]
        out = torch.cat(pools, 1)
        logit = self.fc(out)

        return torch.sigmoid(logit)

