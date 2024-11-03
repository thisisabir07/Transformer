import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Any


class LanguageDataset(Dataset):
    def __init__(
        self,
        dataset,
        source_tokenizer,
        target_tokenizer,
        source_language,
        target_language,
        sequence_length,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_language = source_language
        self.target_language = target_language
        self.sequence_length = sequence_length

        self.sos_token = torch.tensor(
            [source_tokenizer.token_to_id(["[SOS]"])], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [source_tokenizer.token_to_id(["[SOS]"])], dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [source_tokenizer.token_to_id(["[SOS]"])], dtype=torch.int64
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: Any) -> Any:
        source_target_pair = self.dataset[index]
        source_text = source_target_pair["translation"][self.source_language]
        target_text = source_target_pair["translation"][self.target_language]

        encoder_input_tokens = self.source_tokenizer.encode(source_text)
        decoder_input_tokens = self.target_tokenizer.decode(target_text)

        encoder_numeric_padding_tokens = (
            self.sequence_length - len(encoder_input_tokens) - 2
        )
        decoder_numeric_padding_tokens = (
            self.sequence_length - len(decoder_input_tokens) - 1
        )

        if encoder_numeric_padding_tokens < 0 or decoder_numeric_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encoder_input_tokens),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * encoder_numeric_padding_tokens, dtype=torch.int64
                ),
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * decoder_numeric_padding_tokens),
            ]
        )

        label = torch.cat(
            [
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * decoder_numeric_padding_tokens, dtype=torch.int64
                ),
            ]
        )

        assert encoder_input.size(0) == self.sequence_length
        assert decoder_input.size(0) == self.sequence_length
        assert label.size(0) == self.sequence_length

        return {
            "encoder input": encoder_input,
            "decoder input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),  # dimesions: (1,1,sequence_length)
            "decoder_mask": (decoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int()
            & causal_mask(
                decoder_input.size(0)
            ),  # dimensions: (1, sequence_length) & (1, sequence_length, sequence_length)
        }


def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
