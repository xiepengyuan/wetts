# -*- coding: utf-8 -*-

from wetts.vits.text import symbols_mix


def gen_mix_phone_table():
    zh_phone_table_path = "/data1/xiepengyuan/workspace/audio/wetts/examples/wangwen/data/wangwen/phones.txt"
    mix_phone_table_path = "/data1/xiepengyuan/workspace/audio/wetts/examples/ljspeech-mix/data/ljspeech-mix/phones.txt"
    new_lines = []
    with open(zh_phone_table_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            _, index = line.split()
            new_lines.append(line)
        index = int(index)
    for symbol in symbols_mix:
        index += 1
        new_lines.append(f"{symbol} {str(index)}")
    with open(mix_phone_table_path, "w") as f:
        for line in new_lines:
            f.write(f"{line}\n")


if __name__ == '__main__':
    gen_mix_phone_table()
