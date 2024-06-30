# -*- coding: utf-8 -*-

import json
import os
import time
from tqdm import tqdm
from vision_demo import upload_file


def upload_audio():
    speaker = "molinger-moyin"
    date = "20230815"

    test_dir = "/data1/xiepengyuan/workspace/audio/wetts/test"
    json_path = os.path.join(test_dir, "results", date, "json", f"{speaker}.json")
    new_json_path = os.path.join(test_dir, "results", date, "json", f"{speaker}_s3.json")
    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
        for item in tqdm(items):
            url = upload_file(5, item["audio_path"], 60*60*24*30*6, "tts_test")
            item["audio_url"] = url
            time.sleep(0.1)
    with open(new_json_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    upload_audio()
