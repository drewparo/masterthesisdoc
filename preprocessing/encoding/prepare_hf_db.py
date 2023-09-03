# Implementation derived from https://github.com/tloen/alpaca-lora
# Andrea Parolin @drewparo
# Prepare any code dataset from HF for finetuning.

import sys
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch
import requests
import json
from torch.utils.data import random_split
from lit_llama.tokenizer import Tokenizer
from tqdm import tqdm
from datasets import load_dataset
import gc

IGNORE_INDEX = -1

DATA_FILE_NAME = "input.txt"


def prepare(
        destination_path: Path = Path("data/any"),
        tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
        test_split_ratio: float = 0.1,  # default 90% train, 10% validation
        max_seq_length: int = 8196,
        seed: int = 42,
        hf_dataset_name: str = "drewparo/bigquery-swift-filtered-no-duplicate",
        hf_code_column_name: str = "content",
) -> None:
    """Prepare any hf dataset for finetuning .

    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """

    dataset = load_dataset(hf_dataset_name)['train']
    splitted_db = dataset.train_test_split(test_size=test_split_ratio, seed=seed)
    train_set, test_set = splitted_db['train'][hf_code_column_name], splitted_db['test'][hf_code_column_name]

    destination_path.mkdir(parents=True, exist_ok=True)

    file_path = destination_path
    print('file_path\t' + str(file_path))

    if not file_path.exists():
        raise AssertionError(f"{hf_code_column_name} not valid")

    tokenizer = Tokenizer(tokenizer_path)

    print(f"train has {len(train_set):,} samples")
    print(f"val has {len(test_set):,} samples")

    print("Processing train split ...")
    train_set = [
        prepare_line(line, tokenizer, max_seq_length) for line in tqdm(train_set)
    ]
    torch.save(train_set, file_path / "train.pt")
    del train_set
    gc.collect()
    print("Processing test split ...")
    test_set = [
        prepare_line(line, tokenizer, max_seq_length) for line in tqdm(test_set)
    ]
    torch.save(test_set, file_path / "test.pt")
    del test_set
    gc.collect()


def prepare_line(line: str, tokenizer: Tokenizer, max_length: int):
    """Processes a single sample.

    This function processes the line to produce the tokenized version of it.
    """
    encoded_full_prompt = tokenize(tokenizer, line, max_length=max_length, eos=False)
    return {
        "input_ids": encoded_full_prompt,
        "labels": encoded_full_prompt,
    }


def tokenize(
        tokenizer: Tokenizer, string: str, max_length: int, eos=True
) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=eos, max_length=max_length)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
