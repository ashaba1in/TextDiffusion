import os
import argparse
import torch
from torch import nn

from tqdm.auto import tqdm

from dataset import get_dataloader
from transformers import T5ForConditionalGeneration, T5Config, AutoTokenizer

import wandb

from evaluation import Evaluator
from model import TextDDPM
from trainer import Trainer

wandb.login()


def parse_args():
    parser = argparse.ArgumentParser()
    # DATASET
    parser.add_argument("--data_folder", type=str, default='GYAFC/Family_Relationships', help="Data path")
    parser.add_argument("--models_folder", type=str, default='models', help="Data path")

    parser.add_argument("--labels", type=str, default=('formal', 'informal'), nargs="+", help="Style label names")

    parser.add_argument("--max_text_len", type=int, default=50, help="Maximum length of text")
    parser.add_argument("--encoding_mean", type=float, default=-0.01006, help="Mean value used for normalization")
    parser.add_argument("--encoding_std", type=float, default=0.45312, help="Std value used for normalization")

    # MODEL
    parser.add_argument("--context_mode", type=str, default='context', help="Context information for model")

    parser.add_argument("--context_dim", type=int, default=512, help="Hidden dimension of Context Encoder output")
    parser.add_argument("--embedding_dim", type=int, default=768, help="Hidden dimension of input DDPM encodings")
    parser.add_argument("--hidden_t_dim", type=int, default=128, help="Hidden dimension of timestep encodings")

    parser.add_argument("--ddpm_dim", type=int, default=512, help="Hidden dimension of ddpm")

    # TRAINING
    parser.add_argument("--train_context", action="store_true", help="True if we Context Encoder in trainable")
    parser.add_argument("--noise_scheduler", type=str, default='sqrt', help="Noise scheduler type")
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="Amount of timesteps for diffusion")

    parser.add_argument("--ddpm_lr", type=float, default=1e-4, help="DDPM learning rate")
    parser.add_argument("--ce_lr", type=float, default=1e-4, help="Contex Encoder learning rate")

    parser.add_argument("--n_epochs", type=int, default=200, help="Number of epoches")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")

    parser.add_argument("--eval_epoch_step", type=int, default=10, help="Evaluate every nth epoch")
    parser.add_argument("--evaluation_metrics",
                        type=str, default=('bleu', 'self_bleu', 'style_accuracy'),
                        nargs="+", help="Metrics for evaluation"
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parse_args()
    args.device = device

    is_style_transfer = 'style' in args.context_mode

    train_data_paths = [os.path.join(args.data_folder, f'train/{style}') for style in args.labels]
    train_loader = get_dataloader(train_data_paths, args, shuffle=True)
    val_data_paths = [os.path.join(args.data_folder, f'tune/{style}') for style in args.labels]
    if is_style_transfer:
        val_loaders = [
            get_dataloader(val_data_paths[:1], args, shuffle=False),
            get_dataloader(val_data_paths[1:], args, shuffle=False)
        ]
    else:
        val_loaders = [get_dataloader(val_data_paths, args, shuffle=False)]

    t5_config = T5Config(d_model=args.ddpm_dim, d_ff=args.ddpm_dim * 4)
    ddpm = T5ForConditionalGeneration(t5_config).decoder.to(device)

    if args.train_context:
        context_encoder = T5ForConditionalGeneration(t5_config).encoder.to(device)
    else:
        context_encoder = T5ForConditionalGeneration.from_pretrained("t5-small").encoder.to(device)
    context_tokenizer = AutoTokenizer.from_pretrained("t5-small")

    text_ddpm = TextDDPM(args, ddpm, context_encoder=context_encoder)

    trainer = Trainer(args, text_ddpm, context_tokenizer=context_tokenizer)
    evaluator = Evaluator(args, trainer, val_loaders)
    print(args)

    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    name = f'text_diffusion_{slurm_job_id}'

    wandb_config = {
        'context_mode': args.context_mode,
        'noise_scheduler': args.noise_scheduler,
        'batch_size': args.batch_size,
        'evaluation_metrics': args.evaluation_metrics,
        'train_context': args.train_context,
        'ddpm_dim': args.ddpm_dim,
        'labels': args.labels
    }
    wandb.init(
        project='text_duffusion',
        name=name,
        config=wandb_config
    )

    for e in tqdm(range(args.n_epochs)):
        trainer.train_one_epoch(
            train_loader
        )

        if e % args.eval_epoch_step == 0:
            results = {}
            if is_style_transfer:
                for style_label in range(2):
                    style_name = train_loader.dataset.idx_to_label[style_label]
                    eval_results = evaluator.evaluate(
                        text_ddpm,
                        style_label=style_label,
                        predictions_path=f'logs/{name}_epoch{e}_{style_name}'
                    )
                    for key, value in eval_results.items():
                        results[f'{key}_{style_name}'] = value
            else:
                eval_results = evaluator.evaluate(
                    text_ddpm,
                    predictions_path=f'logs/{name}_epoch{e}'
                )
                results = eval_results

            wandb.log(results)

        if e % 50 == 49:
            trainer.ddpm.save(os.path.join(args.models_folder, f'{name}.pt'))

    trainer.ddpm.save(os.path.join(args.models_folder, f'{name}.pt'))
