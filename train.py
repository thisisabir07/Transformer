import torch
import torch.nn as nn

from datasets import load_dataset
from tokenizers import Token, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path


def get_or_build_tokenizer(config, dataset, language):
    tokenizer_path = Path(config["tokenizer_file"].format(language))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2,
            show_progress=True,
        )
        tokenizer.train_from_iterator(
            get_all_sentences(dataset, language), trainer=trainer
        )
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_dataset(config):
    raw_dataset = load_dataset(
        "opus_books",
        f'{config["source_language"]}-{config["target_language"]}',
        split="train",
    )

    source_tokenizer = get_or_build_tokenizer(
        config, raw_dataset, config["source_language"]
    )
    target_tokenizer = get_or_build_tokenizer(
        config, raw_dataset, config["target_language"]
    )

    train_dataset_size = int(0.9 * len(raw_dataset))
    validatiion_dataset_size = len(raw_dataset) - train_dataset_size
    train_dataset_size, validatiion_dataset_size = random_split(
        raw_dataset, [train_dataset_size, validatiion_dataset_size]
    )
