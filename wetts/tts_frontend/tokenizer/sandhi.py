# -*- coding: utf-8 -*-

import json
import os
import re
from collections import defaultdict
from itertools import zip_longest

from wetts.tts_frontend.tokenizer.constants import word33_counter


def fix_last_yi_bu(sentence):
    """句尾“一不”不变调"""
    for char1, char2 in zip_longest(sentence.chars, sentence.chars[1:]):
        if (char1.phoneme) and (char2 is None or char2.phoneme is None):
            if char1.text == "不":
                char1.phoneme = re.sub(r"[1234]$", "4", char1.phoneme)
                char1.phoneme_changeable = False
            if char1.text == "一":
                char1.phoneme = re.sub(r"[1234]$", "1", char1.phoneme)
                char1.phoneme_changeable = False


def fix_yi_bu(sentence):
    """一不变调，从后往前判断"""
    chars_rev = list(reversed(sentence.chars))
    for char1, char2 in zip(chars_rev[1:], chars_rev[:-1]):
        if char1.phoneme and char2.phoneme:
            if char1.text == "不" and char2.phoneme[-1] == "4":
                char1.phoneme = re.sub(r"\d$", "2", char1.phoneme)
            # “一”字，先由模型判断变不变调，变调标为4
            if char1.text == "一" and char1.phoneme[-1] == "4":
                if char2.phoneme[-1] == "4":
                    char1.phoneme = re.sub(r"\d$", "2", char1.phoneme)
                elif char2.phoneme[-1] in {"1", "2", "3"}:
                    char1.phoneme = re.sub(r"\d$", "4", char1.phoneme)

    return sentence


def fix_33(sentence):
    p1_words = []
    p234_words = []
    p1_word = []
    p234_word = []
    for char in sentence.chars:
        p1_word.append(char)
        p234_word.append(char)
        if char.prosody:
            p1_words.append(p1_word)
            p1_word = []
            if re.fullmatch(r"#[234]", char.prosody):
                p234_words.append(p234_word)
                p234_word = []

    for word in p1_words:
        if len(word) == 2:
            if (
                    word[0].phoneme
                    and word[1].phoneme
                    and word[0].phoneme[-1] == "3"
                    and word[1].phoneme[-1] == "3"
            ):
                word[0].phoneme = re.sub(r"3$", "6", word[0].phoneme)
        elif len(word) > 2:
            word_counts = []
            for char1, char2 in zip(word[:-1], word[1:]):
                if (
                        char1.phoneme
                        and char2.phoneme
                        and char1.phoneme[-1] == "3"
                        and char2.phoneme[-1] == "3"
                ):
                    word_text = char1.text + char2.text
                    count = word33_counter.get(word_text, 0)
                    word_counts.append((char1, char2, count))
            word_counts = sorted(word_counts, key=lambda x: x[2], reverse=True)
            for word_count in word_counts:
                if (
                        word_count[0].phoneme[-1] == "3"
                        and word_count[1].phoneme[-1] == "3"
                ):
                    word_count[0].phoneme = re.sub(r"3$", "6", word_count[0].phoneme)

    for word in p234_words:
        if len(word) == 2:
            if (
                    word[0].phoneme
                    and word[1].phoneme
                    and word[0].phoneme[-1] == "3"
                    and word[1].phoneme[-1] == "3"
            ):
                word[0].phoneme = re.sub(r"3$", "6", word[0].phoneme)
        elif len(word) > 2:
            word_counts = []
            for char1, char2 in zip(word[:-1], word[1:]):
                if (
                        char1.phoneme
                        and char2.phoneme
                        and char1.phoneme[-1] == "3"
                        and char2.phoneme[-1] == "3"
                ):
                    word_text = char1.text + char2.text
                    count = word33_counter.get(word_text, 0)
                    word_counts.append((char1, char2, count))
            word_counts = sorted(word_counts, key=lambda x: x[2], reverse=True)
            for word_count in word_counts:
                if (
                        word_count[0].phoneme[-1] == "3"
                        and word_count[1].phoneme[-1] == "3"
                ):
                    word_count[0].phoneme = re.sub(r"3$", "6", word_count[0].phoneme)

    for char in sentence.chars:
        if char.phoneme and char.phoneme[-1] == "6":
            char.phoneme = re.sub(r"6$", "2", char.phoneme)

    return sentence


def create_word33_counter(file_path, out_path):
    from constants import pinyin_dict

    chars3 = set()
    for char in pinyin_dict:
        for pinyin in pinyin_dict[char]:
            if pinyin[-1] == "3":
                chars3.add(char)
    word33_counter = defaultdict(int)
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            for char1, char2 in zip(line[:-1], line[1:]):
                if (char1 in chars3) and (char2 in chars3):
                    word = char1 + char2
                    word33_counter[word] += 1
    new_word33_counter = {
        word: word33_counter[word]
        for word in word33_counter
        if word33_counter[word] > 1
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(new_word33_counter, f, ensure_ascii=False, indent=2)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    file_path = "/data1/jiesen/datasets/zhwiki-latest-pages-articles.xml"
    out_path = os.path.join(here, "data", "word33.json")
    create_word33_counter(file_path, out_path)


if __name__ == "__main__":
    main()
