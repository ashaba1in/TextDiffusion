import os
from typing import List

from transformers import BertTokenizer, BertForMaskedLM, BertConfig

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


def get_dataloader(data_paths, args, shuffle=False):
    dataset = TextDataset(
        data_paths=data_paths,
        args=args,
        has_labels=('style' in args.context_mode),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=shuffle,
        num_workers=1,
        collate_fn=partial(dataset.collate_pad, cutoff=args.max_text_len),
    )

    return dataloader

    # while True:
    #     for batch in dataloader:
    #         yield batch


class TextDataset(Dataset):
    def __init__(
            self,
            data_paths: List[str],
            args,
            has_labels: bool = False,
    ) -> None:
        super().__init__()
        self.args = args

        self.label_to_idx = None
        self.data_paths = data_paths
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.encoder = BertForMaskedLM(BertConfig())
        self.encoder.load_state_dict(torch.load(os.path.join(args.models_folder, 'bert_ae.pt')))
        self.encoder = self.encoder.bert.to(args.device)

        self.read_data()
        if has_labels:
            self.read_labels()

        # mean and std for encoding normalization
        self.mean = args.encoding_mean
        self.std = args.encoding_std

    def read_data(self):
        logging.info("Reading data from {}".format(self.data_paths))
        data = []
        for data_path in self.data_paths:
            data.extend(
                pd.read_csv(
                    data_path, sep="\t", header=None
                )[0].apply(lambda x: x.strip()).tolist()
            )  # read text file

        logging.info(f"Tokenizing {len(data)} sentences")

        self.text = data

        # check if tokenizer has a method 'encode_batch'
        if hasattr(self.tokenizer, 'encode_batch'):
            encoded_input = self.tokenizer.encode_batch(self.text)
            self.input_ids = [x.ids for x in encoded_input]
        else:
            encoded_input = self.tokenizer(self.text)
            self.input_ids = encoded_input["input_ids"]

    def read_labels(self):
        self.labels = []
        for data_path in self.data_paths:
            self.labels.extend(pd.read_csv(data_path, sep="\t", header=None)[1].tolist())

        # check if labels are already numerical
        self.labels = [str(x) for x in self.labels]
        if isinstance(self.labels[0], int):
            return
        # if not, convert to numerical
        all_labels = self.args.labels
        self.label_to_idx = {label: i for i, label in enumerate(all_labels)}
        self.idx_to_label = {i: label for i, label in enumerate(all_labels)}
        self.labels = [self.label_to_idx[label] for label in self.labels]

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, i):
        out_dict = {
            "text": self.text[i],
            "input_ids": self.input_ids[i],
        }
        if hasattr(self, "labels"):
            out_dict["label"] = self.labels[i]
        return out_dict

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean

    def collate_pad(self, batch, cutoff: int):
        max_token_len = 0
        num_elems = len(batch)

        for i in range(num_elems):
            max_token_len = max(max_token_len, len(batch[i]["input_ids"]))

        max_token_len = min(cutoff, max_token_len)

        tokens = torch.zeros(num_elems, max_token_len).long()
        tokens_mask = torch.zeros(num_elems, max_token_len).long()

        has_labels = False
        if "label" in batch[0]:
            labels = torch.zeros(num_elems).long()
            has_labels = True

        for i in range(num_elems):
            toks = batch[i]["input_ids"][:max_token_len]
            length = len(toks)
            tokens[i, :length] = torch.LongTensor(toks)
            tokens_mask[i, :length] = 1
            if has_labels:
                labels[i] = batch[i]["label"]

        text = [batch[i]["text"] for i in range(num_elems)]

        device = next(self.encoder.parameters()).device
        with torch.inference_mode():
            encodings = self.encoder(
                input_ids=tokens.to(device),
                attention_mask=tokens_mask.to(device)
            )['last_hidden_state'].cpu()
            encodings = self.normalize(encodings)

        if has_labels:
            return {
                "text": text,
                "input_ids": tokens,
                "attention_mask": tokens_mask,
                "encodings": encodings,
                "labels": labels,
            }
        else:
            return {
                "text": text,
                "input_ids": tokens,
                "attention_mask": tokens_mask,
                "encodings": encodings,
            }
