# -*- coding: utf-8 -*-

from wetts.vits.text import letters


def reformat_en_label():
    label_path = "/data1/xiepengyuan/workspace/audio/wetts/examples/ljspeech-mix_v2/data/ljspeech-mix_v2/010001-020000.txt.cleaned"
    dst_label_path = "/data1/xiepengyuan/workspace/audio/wetts/examples/ljspeech-mix_v2/data/ljspeech-mix_v2/en_all.txt"
    new_lines = []
    with open(label_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            path, text = line.split("|")
            words = text.split(" ")
            phonemes = ["sil"]
            try:
                for i, word in enumerate(words):
                    if i == len(words) - 1:
                        if word[-1] in ";:,.!?¡¿—…":
                            word = word[:-1]
                        prosody_flag = "#4"
                    else:
                        if word == "":
                            continue
                        if word[-1] in ";:,.!?¡¿—…":
                            word = word[:-1]
                            prosody_flag = "#3"
                        else:
                            prosody_flag = "#0"
                    for char in word:
                        if char in letters:
                            char = "@" + char
                        phonemes.append(char)
                    phonemes.append(prosody_flag)
                new_text = " ".join(phonemes)
                new_line = f"{path}|{new_text}"
                new_lines.append(new_line)
            except Exception as e:
                print("error: ", path, words)
                break
    with open(dst_label_path, "w") as f:
        for new_line in new_lines:
            f.write(new_line + "\n")


if __name__ == '__main__':
    reformat_en_label()
