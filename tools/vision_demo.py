# -*- coding: utf-8 -*-
import requests

VISION_URL = "http://innervision.cc.163.com/inner/upload/token/"


def upload_file(_type, file_path, ttl=60*60*24*30*3, usage="tts"):
    # 请自行处理http请求异常
    res = requests.get(VISION_URL, params={"type": _type, "ttl": ttl, "usage": usage})
    if res.status_code != 200:
        return None
    res.raise_for_status()
    data = res.json()
    url = data.get("url", "")
    headers = data.get("headers", None)
    with open(file_path, 'rb') as f:
        data = f.read()
        res = requests.post(url, data=data, headers=headers)
        if res.status_code != 200:
            return None
        res.raise_for_status()
        return res.json()["url"]


def run():
    url = upload_file(5, "/data1/xiepengyuan/audio/sv/wave/liqiang/liqiang_005/liqiang_005_0001.wav", ttl=None, usage="speaker_verification")
    print(url)


if __name__ == "__main__":
    run()
