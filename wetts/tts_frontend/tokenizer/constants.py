import os
import re
import json

from collections import ChainMap
from wetts.tts_frontend.tokenizer.phoneme_dataset import read_counter

this_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(this_dir, "data")
pinyin_dict_path = os.path.join(data_dir, "pinyin_dict.txt")
phrases_dict_path = os.path.join(data_dir, "phrases_dict.txt")
phrases_dict2_path = os.path.join(data_dir, "phrases_dict2.txt")
phrases_dict3_path = os.path.join(data_dir, "phrases_dict3.txt")
word33_counter_path = os.path.join(data_dir, "word33.json")
alpha2pinyin_path = os.path.join(data_dir, "alpha2pinyin.txt")
polyphone_counter_path = os.path.join(data_dir, "polyphone_counter.json")


def read_pinyin_dict(file_path):
    pinyin_dict = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            line = re.sub(r"\s*#.*$", "", line)
            if re.fullmatch(r"", line):
                continue
            char, pinyin_str = re.split(r"\s+", line, maxsplit=1)
            pinyins = re.split(r"\s+", pinyin_str)
            pinyin_dict[char] = pinyins
    return pinyin_dict


def read_phrases_dict(file_path):
    phrases_dict = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            line = re.sub(r"\s*#.*$", "", line)
            if re.fullmatch(r"", line):
                continue
            phrase, pinyin_str = re.split(r"\s+", line, maxsplit=1)
            pinyins = re.split(r"\s+", pinyin_str)
            phrases_dict[phrase] = pinyins
    return phrases_dict


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        out_dict = json.load(f)
    return out_dict


pinyin_dict = read_pinyin_dict(pinyin_dict_path)
phrases_dict = read_phrases_dict(phrases_dict_path)
phrases_dict2 = read_phrases_dict(phrases_dict2_path)
phrases_dict3 = read_phrases_dict(phrases_dict3_path)
all_phrases_dict = ChainMap(phrases_dict3, phrases_dict, phrases_dict2)
word33_counter = read_json(word33_counter_path)
alpha2pinyin = read_phrases_dict(alpha2pinyin_path)
polyphone_counter = read_counter(polyphone_counter_path)
