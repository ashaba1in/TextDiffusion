import os
import argparse
import torch
from torch import nn

from tqdm.auto import tqdm

from dataset import get_dataloader, inf_dataloader
from transformers import T5ForConditionalGeneration, T5Config, AutoTokenizer, BertModel, BertConfig, BertTokenizerFast

import wandb

from evaluation import Evaluator
from model import TextDDPM
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    # DATASET
    parser.add_argument("--dataset", type=str, default='gyafc', help="Dataset")
    # parser.add_argument("--data_folder", type=str, default='datasets/GYAFC/Family_Relationships', help="Data path")
    parser.add_argument("--models_folder", type=str, default='models', help="Data path")

    parser.add_argument("--labels", type=str, default=[], nargs="+", help="Style label names")

    parser.add_argument("--max_text_len", type=int, default=32, help="Maximum length of text")

    # MODEL
    parser.add_argument("--context_mode", type=str, default='context', help="Context information for model")

    parser.add_argument("--context_dim", type=int, default=768, help="Hidden dimension of Context Encoder output")
    parser.add_argument("--embedding_dim", type=int, default=768, help="Hidden dimension of input DDPM encodings")
    parser.add_argument("--hidden_t_dim", type=int, default=768, help="Hidden dimension of timestep encodings")

    parser.add_argument("--ddpm_dim", type=int, default=768, help="Hidden dimension of ddpm")
    parser.add_argument("--ddpm_num_hidden_layers", type=int, default=12, help="Number of hidden layers of ddpm")

    # TRAINING
    parser.add_argument("--train_context", action="store_true", help="True if we Context Encoder in trainable")
    parser.add_argument("--noise_scheduler", type=str, default='linear', help="Noise scheduler type")
    parser.add_argument("--num_train_timesteps", type=int, default=2000, help="Amount of timesteps for diffusion")

    parser.add_argument("--ddpm_lr", type=float, default=1e-4, help="DDPM learning rate")
    parser.add_argument("--ce_lr", type=float, default=1e-4, help="Contex Encoder learning rate")
    parser.add_argument("--num_warmup", type=int, default=5000, help="Number of warmup steps")

    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--training_iters", type=int, default=350_000, help="Number of training iterations")
    parser.add_argument("--eval_every", type=int, default=10_000, help="Evaluate every n iters")

    parser.add_argument("--mp", action="store_true", help="Mixed precision")

    parser.add_argument("--evaluation_metrics",
                        type=str, default=['bloom'],
                        nargs="+", help="Metrics for evaluation"
    )

    args = parser.parse_args()

    if args.dataset == 'gyafc':
        args.labels = ['formal', 'informal']
    elif args.dataset == 'yelp':
        args.labels = ['positive', 'negative']
    return args


if __name__ == '__main__':
    wandb.login()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parse_args()
    args.device = device

    is_style_transfer = 'style' in args.context_mode

    train_loader = inf_dataloader(get_dataloader('train', args, shuffle=True))
    if is_style_transfer:
        val_loaders = [
            get_dataloader('validation', args, label=label, shuffle=False) for label in args.labels
        ]
    else:
        val_loaders = [get_dataloader('validation', args, shuffle=False)]

    config = BertConfig.from_pretrained('bert-base-uncased')
    config.hidden_size = args.ddpm_dim
    config.num_hidden_layers = args.ddpm_num_hidden_layers
    # config.num_attention_heads = 8
    config.is_decoder = True
    if len(args.context_mode) > 0:
        config.add_cross_attention = True
    ddpm = BertModel(config).encoder

    # t5_config = T5Config(d_model=args.ddpm_dim, d_ff=args.ddpm_dim * 4)
    # ddpm = T5ForConditionalGeneration(t5_config).decoder

    if 'context' in args.context_mode:
        if args.train_context:
            context_encoder = BertModel(BertConfig.from_pretrained("bert-base-uncased"))
            # context_encoder = T5ForConditionalGeneration(t5_config).encoder
        else:
            context_encoder = BertModel.from_pretrained("bert-base-uncased").eval()
            # context_encoder = T5ForConditionalGeneration.from_pretrained("t5-small").encoder
        context_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    else:
        context_encoder = None
        context_tokenizer = None

    text_ddpm = TextDDPM(args, ddpm, context_encoder=context_encoder)

    trainer = Trainer(args, text_ddpm, context_tokenizer=context_tokenizer)
    evaluator = Evaluator(args, trainer, val_loaders)
    print(args)

    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    name = f'{args.dataset}_"{args.context_mode}"{args.mp}mp_{args.batch_size}bs_{args.noise_scheduler}noise_{slurm_job_id}'

    wandb_config = {
        'dataset': args.dataset,
        'context_mode': args.context_mode,
        'num_train_timesteps': args.num_train_timesteps,
        'noise_scheduler': args.noise_scheduler,
        'batch_size': args.batch_size,
        'evaluation_metrics': args.evaluation_metrics,
        'train_context': args.train_context,
        'ddpm_dim': args.ddpm_dim,
        'ddpm_num_hidden_layers': args.ddpm_num_hidden_layers,
        'labels': args.labels,
        'max_text_len': args.max_text_len,
        'mp': args.mp,
        'ddpm_params': sum(p.numel() for p in text_ddpm.ddpm.parameters()),
    }
    wandb.init(
        project='text_duffusion',
        name=name,
        config=wandb_config
    )

    try:
        os.mkdir(f'logs/{slurm_job_id}')
    except:
        pass

    e = 0
    while True:
    # for e in tqdm(range(args.n_epochs)):
        trainer.train_one_epoch(
            train_loader
        )

        results = {}
        if is_style_transfer:
            for style_label in range(2):
                style_name = trainer.idx_to_label[style_label]
                eval_results = evaluator.evaluate(
                    text_ddpm,
                    style_label=style_label,
                    predictions_path=f'logs/{slurm_job_id}/epoch{e}_{style_name}_{name}'
                )
                for key, value in eval_results.items():
                    results[f'{key}_{style_name}'] = value
        else:
            eval_results = evaluator.evaluate(
                text_ddpm,
                predictions_path=f'logs/{slurm_job_id}/epoch{e}_{name}'
            )
            results = eval_results

        wandb.log(results)

        if e % 10 == 9:
            trainer.ddpm.save(os.path.join(args.models_folder, f'{name}.pt'))
        e += 1

    trainer.ddpm.save(os.path.join(args.models_folder, f'{name}.pt'))
