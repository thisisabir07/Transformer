from pathlib import Path

import tokenizers
import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Token, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from torch.utils.tensorboard import SummaryWriter
import warnings

from tqdm import tqdm
from dataset import LanguageDataset
from model import transformerBuilder


def get_all_sentences(dataset, language):
    for item in dataset:
        yield item["translation"][language]


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
        tokenizer.save(str(tokenizer_path))
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
    validation_dataset_size = len(raw_dataset) - train_dataset_size
    train_dataset_raw, validation_dataset_raw = random_split(
        raw_dataset, [train_dataset_size, validation_dataset_size]
    )

    train_dataset = LanguageDataset(
        train_dataset_raw,
        source_tokenizer,
        target_tokenizer,
        config["source_language"],
        config["target_language"],
        config["sequence_length"],
    )

    validation_dataset = LanguageDataset(
        validation_dataset_raw,
        source_tokenizer,
        target_tokenizer,
        config["source_language"],
        config["target_language"],
        config["sequence_length"],
    )

    max_source_length = 0
    max_target_length = 0

    for item in raw_dataset:
        source_ids = source_tokenizer.encode(
            item["translation"][config["source_language"]]
        ).ids
        target_ids = source_tokenizer.encode(
            item["translation"][config["source_language"]]
        ).ids
        max_source_length = max(max_source_length, len(source_ids))
        max_target_length = max(max_target_length, len(target_ids))

    print(f"Max length of source sentence: {max_source_length}")
    print(f"Max length of target sentence: {max_target_length}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)


def get_model(config, source_vocabulary_length, target_vocabulary_length):
    model = transformerBuilder(
        source_vocabulary_length,
        target_vocabulary_length,
        config["sequence_length"],
        config["sequence_length"],
        config["d_model"],
    )
    return model


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, validation_dataloader, source_tokenizer, target_tokenizer = (
        get_dataset(config)
    )
    model = get_model(
        config, source_tokenizer.get_vocab_size(), target_tokenizer.get_vocab_size()
    ).to(device)

    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(
        model.parameters(), learning_rate=config["learning_rate"], eps=1e-9
    )

    initial_epoch = 0
    global_step = 0

    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading model: {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=source_tokenizer.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch: {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(
                device
            )  # (batch_size, sequence_length)
            decoder_input = batch["decoder_input"].to(
                device
            )  # (batch_size, sequence_length)
            encoder_mask - batch["encoder_mask"].to(
                device
            )  # (batch_size, 1, 1, sequence_length)
            decoder_mask - batch["decoder_mask"].to(
                device
            )  # (batch_size, 1, sequence_length, sequence_length)

            encoder_output = model.encode(
                encoder_input, encoder_mask
            )  # (batch_size, sequence_length, d_model)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )  # (batch_size, sequence_length, d_model)
            projection_mask = model.project(
                decoder_output
            )  # (batch_size, sequence_length, target_vocabulary_size)

            label = batch["label"].to(device)  # (batch_size, sequence_length)

            loss = loss_fn(
                projection_output.view(-1, target_tokenizer.get_vocab_size()),
                label.view(-1),
            )
            batch_iterator.set_post_fix({f"Loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar("train_loss", loss.item(), global_step)
            writer.flush()

            # Backpropogate the loss:
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # save the model at the end of each epoch:

    model_filename = get_weights_file_path(config, f"{epoch:02d}")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state(),
            "global_step": global_step,
        },
        model_filename,
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
