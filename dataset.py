import os
from typing import List

from datasets import load_dataset, load_from_disk
from transformers import BertTokenizerFast, BertForMaskedLM, BertConfig

import logging
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
from functools import partial

logging.basicConfig(level=logging.INFO)

try:
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass


def get_dataloader(split, args, label=None, shuffle=False):
    if args.dataset == 'gyafc':
        dataset = create_gyafc_dataset(split, label=label)
    elif args.dataset == 'yelp':
        dataset = create_yelp_dataset(split)
    elif args.dataset == 'wikipedia':
        dataset = create_wikipedia_dataset(split)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        drop_last=False,
        pin_memory=True,
        num_workers=2,
        shuffle=shuffle,
    )
    return dataloader


def inf_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def create_gyafc_dataset(split, label=None):
    base_path = 'datasets/GYAFC'
    datasets = ['Family_Relationships', 'Entertainment_Music']
    if label is None:
        labels = ['formal', 'informal']
    else:
        labels = [label]

    data_files = []
    for dataset in datasets:
        for label in labels:
            data_files.append(f'{base_path}/{dataset}/{split}/{label}')
    data_files = {split: data_files}

    dt = load_dataset('csv', data_files=data_files, delimiter="\t", column_names=["inputs", "label"], split=split)

    return dt


def create_yelp_dataset(split, label=None):
    base_path = 'datasets/yelp'
    if label is None:
        labels = ['positive', 'negative']
    else:
        labels = [label]

    data_files = {split: [f'{base_path}/{split}/{label}' for label in labels]}
    dt = load_dataset('csv', data_files=data_files, delimiter="\t", column_names=["inputs", "label"], split=split)

    return dt


def create_wikipedia_dataset(split):
    base_path = "datasets/wikipedia"

    dt = load_from_disk(base_path).get(split)
    return dt
