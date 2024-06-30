# -*- coding: utf-8 -*-

import datetime
import math
import os
import pandas as pd


def excel_to_lrc():
    speaker = "kuakua"
    date = "20230815"

    test_dir = "/data1/xiepengyuan/workspace/audio/wetts/test"
    corpus_path = os.path.join(test_dir, "corpus", f"{date}.xlsx")
    json_path = os.path.join(test_dir, "results", date, "json", f"{speaker}.json")
    dst_wave_dir = os.path.join(test_dir, "results", f"{date}/wave/{speaker}")
    os.makedirs(dst_wave_dir, exist_ok=True)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # 读取Excel文件
    df = pd.read_excel(corpus_path)

    # 读取名为'列名'的列数据
    indexes = df["序号"]
    texts = df["文本"]

    seconds = 0
    base_time = datetime.datetime(1900, 1, 1)
    lrc_path = os.path.join("/data1/xiepengyuan/workspace/audio/wetts/test/lrc", f"{date}.lrc")
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


if __name__ == '__main__':
    excel_to_lrc()
