# -*- coding: utf-8 -*-
import os


def gen_zh_batch_data():
    text_batch_dir = "/data1/xiepengyuan/workspace/audio/wetts/data/text_batch_bs1000"
    os.makedirs(text_batch_dir, exist_ok=True)

    batch_size = 1000

    with open("/data1/xiepengyuan/workspace/audio/wetts/examples/enqi/label/baker_text.txt", 'r') as f:
        lines = f.readlines()
        fout = None
        for i, line in enumerate(lines):
            index, text = line.strip().split("\t")
            if i % batch_size == 0:
                if fout:
                    fout.close()
                last_index = str(int(index)+batch_size-1).zfill(6)
                fout = open(os.path.join(text_batch_dir, f"{index}-{last_index}.txt"), "w")
            fout.write(f"{text}\n")


def gen_en_batch_data():
    text_batch_dir = "/data1/xiepengyuan/workspace/audio/wetts/data/en_text_batch_bs1000"
    os.makedirs(text_batch_dir, exist_ok=True)

    batch_size = 1000

    with open("/data1/xiepengyuan/workspace/audio/wetts/examples/ljspeech-mix_v2/data/ljspeech-mix_v2/010001-020000.txt", 'r') as f:
        lines = f.readlines()
        fout = None
        for i, line in enumerate(lines):
            filename, text = line.strip().split("|")
            index = os.path.splitext(filename)[0]
            if i % batch_size == 0:
                if fout:
                    fout.close()
                last_index = str(int(index)+batch_size-1).zfill(6)
                fout = open(os.path.join(text_batch_dir, f"{index}-{last_index}.txt"), "w")
            fout.write(f"{text}\n")


if __name__ == '__main__':
    gen_en_batch_data()
