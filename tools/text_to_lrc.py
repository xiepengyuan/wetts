# -*- coding: utf-8 -*-

import datetime
import math
import os
import glob


def text_to_lrc():
    src_dir_name = "text_batch_bs1000"
    dst_dir_name = "lrc_batch_bs1000"
    src_dir = f"E:\\workspace\\audio\\wetts\\data\\{src_dir_name}"
    dst_dir = f"E:\\workspace\\audio\\wetts\\data\\{dst_dir_name}"
    os.makedirs(dst_dir, exist_ok=True)
    text_paths = glob.glob(os.path.join(src_dir, "*.txt"))
    for text_path in text_paths:
        name = os.path.splitext(os.path.basename(text_path))[0]
        texts = []
        with open(text_path, encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                texts.append(line.strip())

        seconds = 0
        base_time = datetime.datetime(1900, 1, 1)
        lrc_path = os.path.join(dst_dir, f"{name}.lrc")
        with open(lrc_path, "w", encoding="utf-8") as f:
            for text in texts:
                time_diff = datetime.timedelta(seconds=seconds)
                time_obj = base_time + time_diff
                time_str = time_obj.strftime('%H:%M:%S')
                f.write(f"[{time_str}.00]{text}\n")
                if len(text) > 15:
                    seconds += math.ceil(len(text) / 5) + 3
                else:
                    seconds += math.ceil(len(text) / 5) + 2


def en_text_to_lrc():
    src_dir_name = "en_text_batch_bs1000"
    dst_dir_name = "en_lrc_batch_bs1000"
    src_dir = f"F:\\workspace\\audio\\wetts\\data\\{src_dir_name}"
    dst_dir = f"F:\\workspace\\audio\\wetts\\data\\{dst_dir_name}"
    os.makedirs(dst_dir, exist_ok=True)
    text_paths = glob.glob(os.path.join(src_dir, "*.txt"))
    for text_path in text_paths:
        name = os.path.splitext(os.path.basename(text_path))[0]
        texts = []
        with open(text_path, encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                texts.append(line.strip())

        seconds = 0
        base_time = datetime.datetime(1900, 1, 1)
        lrc_path = os.path.join(dst_dir, f"{name}.lrc")
        with open(lrc_path, "w", encoding="utf-8") as f:
            for text in texts:
                time_diff = datetime.timedelta(seconds=seconds)
                time_obj = base_time + time_diff
                time_str = time_obj.strftime('%H:%M:%S')
                f.write(f"[{time_str}.00]{text}\n")
                words = text.split(" ")
                num_words = 0
                for word in words:
                    num_words += len(word.split("-"))
                seconds += math.ceil(num_words / 2.5) + 4


if __name__ == '__main__':
    en_text_to_lrc()
