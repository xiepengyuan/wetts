import re
import unicodedata
from wetts.tts_frontend.tokenizer.constants import (
    all_phrases_dict,
    alpha2pinyin,
    phrases_dict2,
    phrases_dict3,
    pinyin_dict,
)


def get_tokenized_inputs(char_texts, vocab):
    pre_tokenized_inputs = []
    for char_text in char_texts:
        if re.fullmatch(r"[.]{2,}", char_text):
            char_text = "…"
        char_text = re.sub(r"\s+", "，", char_text)
        new_char_texts = []
        for c in char_text:
            if unicodedata.category(c) != "Mn":
                new_char_texts.append(c)
        new_char_text = "".join(new_char_texts) if len(new_char_texts) else "[UNK]"
        if new_char_text not in vocab:
            new_char_text = "[UNK]"
        pre_tokenized_inputs.append(new_char_text)
    return pre_tokenized_inputs


def change_prosody_by_punct(sentence):
    for idx, char in enumerate(sentence.chars):
        if idx == 0:
            continue
        if re.search(r"[、，。：；？！,:;?!…—~]|\W{2,}", char.text):
            for i in range(idx - 1, -1, -1):
                if re.fullmatch(r"\w+", sentence.chars[i].text):
                    if (
                            sentence.chars[i].prosody is None
                            or sentence.chars[i].prosody == "#1"
                    ):
                        sentence.chars[i].prosody = "#3"

                    if re.search(r"[。？！?!]", char.text):
                        sentence.chars[i].prosody = "#4"
                    break

    if len(sentence.chars) and re.fullmatch(r"\w+", sentence.chars[-1].text):
        sentence.chars[-1].prosody = "#4"
    return sentence


def set_prosody(sentence, pred_p1_tag_ids, pred_p2_tag_ids, pred_p3_tag_ids, idx2tag):
    for char, p1, p2, p3 in zip(
            sentence.chars,
            pred_p1_tag_ids[0][1:-1],
            pred_p2_tag_ids[0][1:-1],
            pred_p3_tag_ids[0][1:-1],
    ):
        match_obj = re.fullmatch(r"\W+", char.text)
        if not match_obj:
            p1_tag = idx2tag[p1]
            p2_tag = idx2tag[p2]
            p3_tag = idx2tag[p3]
            if p3_tag in {"E", "S"}:
                char.prosody = "#3"
            elif p2_tag in {"E", "S"}:
                char.prosody = "#2"
            elif p1_tag in {"E", "S"}:
                char.prosody = "#1"
    return sentence


def single_pinyin(sentence, alpha_pinyin=False):
    for char in sentence.chars:
        if char.text in pinyin_dict:
            char.phoneme = pinyin_dict[char.text][0]
            char.phoneme_src = "single"
        elif re.fullmatch(r"[A-Z]+[-']*[A-Z]*|@", char.text):
            if alpha_pinyin:
                phoneme = []
                for char_text in char.text:
                    phoneme.extend(alpha2pinyin.get(char_text, char_text))
                char.phoneme = " ".join(phoneme)
                char.phoneme_src = "single"
    return sentence


def chars_pinyins(chars, pinyins, phoneme_src="phrase", phoneme_changeable=False):
    assert len(chars) == len(pinyins)
    for char, pinyin in zip(chars, pinyins):
        if char.phoneme_changeable:
            char.phoneme = pinyin
            char.phoneme_src = phoneme_src
            if char.text not in "一不":
                char.phoneme_changeable = phoneme_changeable


def phrase_pinyin(sentence, max_len=4):
    # fmm
    remain = sentence.words
    while remain:
        flag = False
        remain_len = len(remain)
        max_idx = min(remain_len, max_len)
        for end_idx in range(max_idx, 0, -1):
            used_words = remain[:end_idx]
            used_text = "".join([word.text for word in used_words])
            # print(used_text)
            if len(used_text) > 1 and used_text in all_phrases_dict:
                if end_idx == 1:
                    flag = True
                else:
                    for word in used_words:
                        if len(word.text) != 1:
                            flag = True
                            break
                    # if flag:
                    #     print(used_text)
                if flag:
                    chars = [char for word in used_words for char in word.chars]
                    if used_text in phrases_dict3:
                        chars_pinyins(
                            chars, phrases_dict3[used_text], phoneme_changeable=False
                        )
                    elif used_text in phrases_dict2:
                        chars_pinyins(
                            chars, phrases_dict2[used_text], phoneme_changeable=True
                        )
                    else:
                        chars_pinyins(
                            chars, all_phrases_dict[used_text], phoneme_changeable=False
                        )
                    remain = remain[end_idx:]
                    break
                else:
                    chars = [char for word in used_words for char in word.chars]
                    if used_text in phrases_dict3:
                        chars_pinyins(
                            chars, phrases_dict3[used_text], phoneme_changeable=False
                        )
                    else:
                        chars_pinyins(
                            chars, all_phrases_dict[used_text], phoneme_changeable=True
                        )

        if not flag:
            remain = remain[1:]


def change_phoneme_by_prosody(sentence):
    for word in sentence.words:
        chars_len = len(word.chars)
        if chars_len <= 1:
            continue
        start_idx = 0
        for idx, char in enumerate(word.chars):
            if char.prosody is not None:
                end_idx = idx + 1
                if end_idx == start_idx + 1:
                    start_idx = end_idx
                    continue
                if (end_idx == chars_len) and (start_idx == 0):
                    break
                used_chars = word.chars[start_idx:end_idx]
                used_text = "".join([char.text for char in used_chars])
                if used_text in phrases_dict2:
                    chars_pinyins(
                        used_chars, phrases_dict2[used_text], phoneme_changeable=True
                    )
                elif used_text in all_phrases_dict:
                    chars_pinyins(
                        used_chars, all_phrases_dict[used_text], phoneme_changeable=False
                    )
                start_idx = end_idx