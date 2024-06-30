import json
import random
import re

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class PhonemeDataset(Dataset):
    def __init__(self, data_file_path, counter_path, is_train=False):
        self.data_file_path = data_file_path
        self.counter_path = counter_path
        self.counter = read_counter(counter_path)
        self.tokens_list, self.polyphone_list = read_data(data_file_path, self.counter)
        # self.tokens_list = self.tokens_list[:100]
        # self.polyphone_list = self.polyphone_list[:100]
        self.is_train = is_train
        self.polyphones = list(self.counter)
        self.char2phonemes = {char: list(self.counter[char]) for char in self.counter}

    def __len__(self):
        return len(self.tokens_list)

    def __getitem__(self, idx):
        tokens = self.tokens_list[idx]
        position_idx, polyphone, phoneme = self.polyphone_list[idx]

        if self.is_train:
            new_tokens = []
            new_position_idx = position_idx
            for idx, token in enumerate(tokens):
                if idx != position_idx and re.fullmatch(r"\W+", token):
                    if random.random() < 0.9:
                        new_tokens.append(token)
                    else:
                        if position_idx > idx:
                            new_position_idx -= 1
                else:
                    new_tokens.append(token)
        else:
            new_tokens = tokens
            new_position_idx = position_idx

        polyphone_idx = self.polyphones.index(polyphone)
        phoneme_idx = self.char2phonemes[polyphone].index(phoneme)
        phoneme_size = len(self.char2phonemes[polyphone])

        sample = {
            "tokens": new_tokens,
            "position_idx": new_position_idx,
            "polyphone_idx": polyphone_idx,
            "phoneme_idx": phoneme_idx,
            "phoneme_size": phoneme_size,
        }
        return sample


def read_counter(counter_file):
    with open(counter_file, "r", encoding="utf-8") as f:
        counter = json.load(f)
    return counter


def read_data(input_file, counter):
    max_len = 128
    tokens_list = []
    polyphone_list = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            line = re.sub(r"\s+", "，", line)
            tokens = re.split(r"(.\{[a-z]+\d\}|\d+|[a-zA-Z]+|.)", line)
            tokens = [token for token in tokens if token]
            chars = []
            idx_char_phoneme = []
            for idx, token in enumerate(tokens):
                if idx >= max_len - 2:
                    break
                if re.fullmatch(r".\{[a-z]+\d\}", token):
                    char = token[0]
                    phoneme = token[2:-1]
                    chars.append(char)
                    if (char in counter) and (phoneme in counter[char]):
                        idx_char_phoneme.append((idx, char, phoneme))
                else:
                    chars.append(token)
            for x in idx_char_phoneme:
                tokens_list.append(chars)
                polyphone_list.append(x)
    return tokens_list, polyphone_list


class CollateFunction(object):
    def __init__(self, pretrained_model_path):
        self.pretrained_model_path = pretrained_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_path)
        self.max_len = 128
        self.vocab = set(self.tokenizer.vocab)

    def __call__(self, batch):
        pre_tokenized_inputs = []
        for x in batch:
            new_tokens = []
            for token in x["tokens"]:
                if token not in self.vocab:
                    token = "[UNK]"
                new_tokens.append(token)
            pre_tokenized_inputs.append(new_tokens)
        position_ids = [x["position_idx"] + 1 for x in batch]
        polyphone_ids = [x["polyphone_idx"] for x in batch]
        phoneme_ids = [x["phoneme_idx"] for x in batch]
        phoneme_sizes = [x["phoneme_size"] for x in batch]

        encoded = self.tokenizer(
            pre_tokenized_inputs,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            is_split_into_words=True,
            return_tensors="pt",
        )
        token_ids = encoded["input_ids"]
        token_type_ids = encoded["token_type_ids"]
        attention_mask = encoded["attention_mask"]

        sample = {
            "token_ids": token_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "position_ids": torch.LongTensor(position_ids),
            "polyphone_ids": torch.LongTensor(polyphone_ids),
            "phoneme_ids": torch.LongTensor(phoneme_ids),
            "phoneme_sizes": torch.LongTensor(phoneme_sizes),
        }
        return sample
