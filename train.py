from pathlib import Path
import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import warnings
from tqdm import tqdm
from dataset import LanguageDataset, causal_mask
from model import transformerBuilder
from config import get_config
from config import get_weights_file_path


def greedy_decode(
    model, source, source_mask, source_tokenizer, target_tokenizer, max_length, device
):
    sos_index = target_tokenizer.token_to_id("[SOS]")
    eos_index = target_tokenizer.token_to_id("[EOS]")

    # precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)

    # Initialize the decoder input with the SOS token
    decoder_input = torch.empty(1, 1).fill_(sos_index).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_length:
            break

        # add mask for decoder input
        decoder_mask = (
            causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        )

        # calculate decoder output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # to get next token
        prob = model.project(out[:, -1])

        # select the token with maximum probability as per Greeedy search
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device),
            ],
            dim=1,
        )

        if next_word == eos_index:
            break

    return decoder_input.squeeze(0)


def run_validation(
    model,
    validation_dataset,
    source_tokenizer,
    target_tokenizer,
    max_length,
    device,
    print_message,
    global_state,
    writer,
    num_examples=2,
):
    model.eval()
    count = 0

    source_text = []
    expected = []
    predicted = []

    console_width = 80

    with torch.no_grad():
        for batch in validation_dataset:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "the batch size must be 1 for validation"

            model_out = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                source_tokenizer,
                target_tokenizer,
                max_length,
                device,
            )

            source_text = batch["source_text"][0]
            target_text = batch["target_text"][0]
            model_out_text = target_tokenizer.decode(model_out.detach().cpu().numpy())

            print_message("-" * console_width)
            print_message(f"SOURCE: {source_text}")
            print_message(f"TARGET: {target_text}")
            print_message(f"PREDICTED: {model_out_text}")

            if count == num_examples:
                break


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
        target_ids = target_tokenizer.encode(  # Fix here to use target tokenizer
            item["translation"][config["target_language"]]
        ).ids
        max_source_length = max(max_source_length, len(source_ids))
        max_target_length = max(max_target_length, len(target_ids))

    print(f"Max length of source sentence: {max_source_length}")
    print(f"Max length of target sentence: {max_target_length}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

    return train_dataloader, validation_dataloader, source_tokenizer, target_tokenizer


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
        model.parameters(), lr=config["learning_rate"], eps=1e-9
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
            model.train()

            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )
            projection_output = model.project(decoder_output)

            label = batch["label"].to(device)

            loss = loss_fn(
                projection_output.view(-1, target_tokenizer.get_vocab_size()),
                label.view(-1),
            )
            batch_iterator.set_postfix({"Loss": f"{loss.item():6.3f}"})

            writer.add_scalar("train_loss", loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            run_validation(
                model,
                validation_dataloader,
                source_tokenizer,
                target_tokenizer,
                config["sequence_length"],
                device,
                lambda message: batch_iterator.write(message),
                global_step,
                writer,
            )

            global_step += 1

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
