# -*- coding: utf-8 -*-

import os
import time

import numpy as np
from scipy.io import wavfile
import torch


from wetts.tts_frontend.tts_frontend_model import TTSFrontendModel
from wetts.tts_frontend.tokenizer.prosody_phoneme_model import ProsodyPhonemeModel
from wetts.tts_frontend.tokenizer.prosody_phoneme_processor import ProsodyPhonemeProcessor
from wetts.tts_frontend.tokenizer.vits_tokenizer import VitsTokenizer
from wetts.vits import utils
from wetts.vits.models import SynthesizerTrn


class VitsInference:
    def __init__(self, resource_dir, checkpoint_path=None, device="cuda"):
        phone_table_path = os.path.join(resource_dir, "tts_frontend", "phones.txt")
        pretrained_model_name_or_path = os.path.join(resource_dir, "tts_frontend",
                                                     "chinese-electra-180g-small-discriminator")
        prosody_phoneme_pretrained_model_path = \
            os.path.join(resource_dir, "tts_frontend", "prosody_phoneme_chinese-electra-180g-small-discriminator.pt")
        pinyin_lexicon_path = os.path.join(resource_dir, "tts_frontend", "pinyin-lexicon.txt")
        cfg_path = os.path.join(resource_dir, "vits", "base.json")
        if not checkpoint_path:
            checkpoint_path = os.path.join(resource_dir, "vits", "G.pth")

        self.device = torch.device(device)

        self.phone_dict = {}
        with open(phone_table_path) as p_f:
            for line in p_f:
                phone_id = line.strip().split()
                self.phone_dict[phone_id[0]] = int(phone_id[1])

        hps = utils.get_hparams_from_file(cfg_path)

        net_g = SynthesizerTrn(
            len(self.phone_dict) + 1,
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=0,
            **hps.model)
        self.net_g = net_g.to(self.device)

        net_g.eval()
        utils.load_checkpoint(checkpoint_path, net_g, None)

        pp_model = ProsodyPhonemeModel(
            tagset_size=5,
            hidden_dim=128,
            pretrained_model_name_or_path=pretrained_model_name_or_path
        ).to(device)

        pp_processor = ProsodyPhonemeProcessor(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            prosody_phoneme_model=pp_model,
            prosody_phoneme_pretrained_model_path=prosody_phoneme_pretrained_model_path,
            device=device
        )
        vits_tokenizer = VitsTokenizer(
            pinyin_lexicon_path=pinyin_lexicon_path,
            phone_table_path=phone_table_path
        )
        self.tts_frontend_model = TTSFrontendModel(
            sentence_processor=pp_processor,
            tokenizer=vits_tokenizer,
            device=device
        )

    def generate_audio(self, text):
        frd_st = time.time()
        tokens = self.tts_frontend_model.parse(text)
        frd_used_time = time.time() - frd_st
        print('used time: {},'.format(frd_used_time))
        with torch.no_grad():
            x = tokens.to(self.device).unsqueeze(0)
            x_length = torch.LongTensor([tokens.size(0)]).to(self.device)
            sid = torch.LongTensor([0]).to(self.device)
            st = time.time()
            audio = self.net_g.infer(
                x,
                x_length,
                sid=sid,
                noise_scale=.667,
                noise_scale_w=0.8,
                length_scale=1)[0][0, 0].data.cpu().float().numpy()
            print(np.max(np.abs(audio)))
            audio *= 32767 / max(0.01, np.max(np.abs(audio))) * 0.6
            vits_used_time = time.time() - st
            print('used time: {}, RTF {}'.format(vits_used_time,
                                                 vits_used_time / (audio.shape[0] / 22050)))
            audio = np.clip(audio, -32767.0, 32767.0)
        return audio


def main():
    out_dir = "/data1/xiepengyuan/workspace/audio/wetts/examples/enqi/output"
    resource_dir = "/data1/xiepengyuan/workspace/audio/wetts/resources/v1"
    vits_inference = VitsInference(resource_dir)

    texts = [
        "五二零到啦，哥哥，感谢一直以来对我的支持呀，有你的每个夜晚，就像星星一样，点亮了我的直播间。未来希望我们还可以一起长长久久，期待一起去创造更多的浪漫回忆。我要岁岁平安，我要风生水起，但我更要年年有你。",
    ]

    for text in texts:
        audio = vits_inference.generate_audio(text)
        text_name = text[:30] + "..." if len(text) > 30 else text
        audio_path = os.path.join(out_dir, f"{text_name}.wav")
        wavfile.write(audio_path, 22050, audio.astype(np.int16))


def test_checkpoint(device="cuda"):
    speaker = "tongtong"
    checkpoint_step = 350000
    out_dir = f"/data1/xiepengyuan/workspace/audio/wetts/examples/{speaker}/output"
    os.makedirs(out_dir, exist_ok=True)
    resource_dir = "/data1/xiepengyuan/workspace/audio/wetts/resources/v1"
    start_step = checkpoint_step
    end_step = checkpoint_step
    for step in range(start_step, end_step+1, 100000):
        checkpoint_path = f"/data1/xiepengyuan/workspace/audio/wetts/examples/{speaker}/exp/{speaker}/G_{step}.pth"
        print(checkpoint_path)
        vits_inference = VitsInference(resource_dir, checkpoint_path, device=device)
        texts = [
            "早上好啊,昨晚睡得好吗?我睡得超级香,做了好多好梦,现在精神满满!",
            "天气真好,我们出去逛街吧!我看上一条超可爱的连衣裙,你帮我挑一挑看行不行。",
            "啊,就是不想起床,被窝里太舒服了。再睡会儿可以吗?就一小时,我保证不会迟到的。",
            "饿了,我们出去吃什么好呢?我最近在节食,你推荐点不高卡不高脂的吧,最好还超级好吃。",
            "我的新发型怎么样?是不是可爱极了?妈妈说这样子我看起来年轻许多。你也这么觉得吗?",
            "电影真的超级无聊,我都差点睡着了。下次我们选个喜剧片看吧,爱情片实在太无聊了。",
            "快来看,这里的日落真美。天空都是粉红色的,美极了。你快拍张照片发朋友圈。",
            "我要去买张演唱会票,去年错过了真的超级遗憾。这次一定要去,你一定要陪我去!",
            "我的生日快到了,你准备送我什么礼物啊?越贵重越好,开个玩笑的,你送的我都很喜欢。 ",
            "要下雨了,带伞了吗?不然我们去买把伞吧,我看上一把超可爱的,我们出双入对用也很配。",
            "欢迎大家来到恩七的直播间，大家可以在公屏打字和我聊天喔。"
        ]
        texts = [
            # "而Treyarch工作室的系列新作则会向后延期。"
            # "abcdefghijklmnopqrstuvwxyz"
            "今天，《使命召唤》的官方账号发推提出一个问题：“《使命召唤19：现代战争2》的角色、武器和捆绑包是否应该继承到《使命召唤2023》中？”结合之前的爆料，几乎能确认今年秋季会推出的2023年的《使命召唤》为大锤工作室的《使命召唤：现代战争III》，而Treyarch工作室的系列新作则会向后延期。此前有爆料称2024年的《使命召唤》游戏将是一款由Treyarch开发的以海湾战争为背景的全新《黑色行动》系列作品。",
            "曾经，有一份真挚的感情摆在我的面前，我没有好好珍惜，等我失去的时候才追悔莫及，人间最痛苦的事莫过于此。",
            "我好想亲自去试玩一下《黑神话：悟空》啊！大家对于这个试玩会有没有什么期待呢？你们觉得这次的试玩会会给我们带来什么样的游戏内容呢？",
            # "hello, 我的名字是haiqi，今天好happy呀"
        ]
        texts = [
            "天气真好，我们出去逛街吧！",
        ]
        for text in texts:
            audio = vits_inference.generate_audio(text)
            text_name = text[:30] + "..." if len(text) > 30 else text
            audio_path = os.path.join(out_dir, f"{speaker}_{step}_{text_name}.wav")
            wavfile.write(audio_path, 22050, audio.astype(np.int16))


def gen_result():
    import pandas as pd
    import json

    speaker = "kuakua"
    step = 300000
    date = "20230815"

    resource_dir = "/data1/xiepengyuan/workspace/audio/wetts/resources/v1"
    checkpoint_path = f"/data1/xiepengyuan/workspace/audio/wetts/examples/{speaker}/exp/{speaker}/G_{step}.pth"
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

    vits_inference = VitsInference(resource_dir, checkpoint_path)

    items = []

    for index, text in zip(indexes, texts):
        audio = vits_inference.generate_audio(text)
        index = str(int(index)).zfill(3)
        audio_path = os.path.join(dst_wave_dir, f"{speaker}_{step}_{index}.wav")
        wavfile.write(audio_path, 22050, audio.astype(np.int16))
        item = {
            "speaker": speaker,
            "index": index,
            "text": text,
            "audio_path": audio_path
        }
        items.append(item)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    test_checkpoint(device="cpu")
