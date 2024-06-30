# -*- coding: utf-8 -*-

import requests
import json
import os
import pandas as pd


def run():
    speaker = "emotion_female01"
    date = "20230815"

    url = "https://tts-gzailab.nie.netease.com"
    token = 'ec4e039711da9125fb6c6eb3b5febde859f8056e'

    # test_dir = "/data1/xiepengyuan/workspace/audio/wetts/test"
    test_dir = "E:\\SpeechData\\cc\\tts\\test"
    corpus_path = os.path.join(test_dir, "corpus", f"{date}.xlsx")
    json_path = os.path.join(test_dir, "results", date, "json", f"{speaker}_s3.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # 读取Excel文件
    df = pd.read_excel(corpus_path)

    # 读取名为'列名'的列数据
    indexes = df["序号"]
    texts = df["文本"]

    items = []

    for index, text in zip(indexes, texts):
        index = str(index).zfill(3)
        print(text)
        data_json = {
            "token": token,
            'language_type': 'emotion_v2',  # 中英情绪模式
            'tts_model_type': 'HG',
            'text': text,  # 测试文本
            'person_type': speaker,  # 可选音色见网页说明表3
            'wav_speed': 1,  # 语速调节参数：0.5到1.5
            'wav_pitch': 0,  # 音高调节参数：-1000到1000
            'wav_volume': 1,  # 音量调节参数：0.1到5
            'audio_effect': 0,  # 音效设置：0：无音效，1：机器人，2：汤姆猫
            'emotion_type': 'auto',  # 情绪类型设置：auto/normal/serious/anger/happiness/sadness/fear/disgust
            'emotion_scale': 1,  # 情绪强度设置：0-2，默认为1
            'data_type': 'url',  # 语音数据获取方式：url/raw, url需要二次请求，raw直接返回语音数据
            'sample_rate': 22050,  # 采样率，(8000, 16000, 22050, 44100)
            "need_cache": False,
        }
        try:
            req = requests.post(url + '/tts', data=json.dumps(data_json))
            response = json.loads(req.text)
            audio_url = url+response["audio_url"]
            item = {
                "speaker": speaker,
                "index": index,
                "text": text,
                "audio_url": audio_url
            }
            items.append(item)
        except Exception as e:
            print(e)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    run()
