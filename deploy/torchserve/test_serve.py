# -*- coding: utf-8 -*-

import time
import requests
import multiprocessing
import json
import numpy as np
import soundfile as sf
import os
import base64


def post(text, speaker_name, sample_rate):
    data = {"text": text, "speaker_name": speaker_name, "sample_rate": sample_rate, "if_upload": 1, "if_audio": 0}
    data = json.dumps(data, ensure_ascii=False).encode("utf-8")
    result = requests.post("http://localhost:8701/predictions/enqi-vits", data=data, timeout=3000)
    return result


def multi_post(texts, voice_names):
    pool = multiprocessing.Pool(len(texts))
    results = []
    audios = []
    for text, voice_name in zip(texts, voice_names):
        results.append(pool.apply_async(post, args=[text, voice_name]))
    pool.close()
    pool.join()
    for i, result in enumerate(results):
        data = result.get().content
        data = json.loads(data)
        # print(data["state"])
        audio = base64.b64decode(data["audio"])
        audio = np.frombuffer(audio, dtype=np.int16)
        audio = audio / float(1 << 15)
        audios.append(audio)
    return audios


def test():
    # 存放音频目录
    dst_dir = "/data1/xiepengyuan/samples"
    os.makedirs(dst_dir, exist_ok=True)

    speaker_name = "hutao"  # ("hutao", "yebiezhi", "taoxing")
    sample_rate = 22050
    texts = [
        "欢迎澜雨哥哥来到默默的直播间，喜欢主播的话可以点个关注哟！",
    ] * 1
    speaker_names = [speaker_name for i in range(len(texts))]
    sample_rates = [sample_rate for i in range(len(texts))]

    start_time = time.time()
    audios = multi_post(texts, speaker_names)
    end_time = time.time()
    for audio, text, voice_name in zip(audios, texts, voice_names):
        sf.write(os.path.join(dst_dir, f"{voice_name}_{text}.wav"), audio, samplerate=22050)
    print(f"used time: {end_time-start_time}, num_texts: {len(texts)}, state: {len(audios)==len(texts)}")


def run():
    # 存放音频目录
    dst_dir = "/data1/xiepengyuan/samples"
    os.makedirs(dst_dir, exist_ok=True)
    text = "五二零到啦，哥哥，感谢一直以来对我的支持呀，有你的每个夜晚，就像星星一样，点亮了我的直播间。未来希望我们还可以一起长长久久，期待一起去创造更多的浪漫回忆。我要岁岁平安，我要风生水起，但我更要年年有你。"
    speaker_name = "enqi"
    sample_rate = 16000
    result = post(text, speaker_name, sample_rate)
    data = result.json()
    audio = base64.b64decode(data["audio"])
    print(data["audio_url"])
    audio = np.frombuffer(audio, dtype=np.int16)
    audio = audio / float(1 << 15)
    sf.write(os.path.join(dst_dir, f"{speaker_name}_{text}.wav"), audio, samplerate=sample_rate)


if __name__ == '__main__':
    run()
