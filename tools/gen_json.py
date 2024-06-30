# -*- coding: utf-8 -*-

import json
import os
import glob
import shutil


def gen_result():
    """
    用于从其他地方生成的音频
    """
    speaker = "molinger-moyin"
    org_speaker = "wangwen"
    step = "0"
    date = "20230815"

    test_dir = "/data1/xiepengyuan/workspace/audio/wetts/test"
    src_json_path = os.path.join(test_dir, "results", date, "json", f"{org_speaker}.json")
    json_path = os.path.join(test_dir, "results", date, "json", f"{speaker}.json")
    dst_wave_dir = os.path.join(test_dir, "results", f"{date}/wave/{speaker}")
    os.makedirs(dst_wave_dir, exist_ok=True)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(src_json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    indexes = [item["index"] for item in items]
    audio_paths = [item["audio_path"] for item in items]
    texts = [item["text"] for item in items]

    target_audio_paths = glob.glob(os.path.join(dst_wave_dir, "*.wav"))
    target_audio_paths.sort()

    items = []

    for i, (index, text, audio_path, target_audio_path) in enumerate(zip(indexes, texts, audio_paths, target_audio_paths)):
        dst_audio_path = os.path.join(dst_wave_dir, f"{speaker}_{step}_{index}.wav")
        shutil.move(target_audio_path, dst_audio_path)
        item = {
            "speaker": speaker,
            "index": index,
            "text": text,
            "audio_path": dst_audio_path
        }
        items.append(item)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    gen_result()
