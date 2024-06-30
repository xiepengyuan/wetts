# -*- coding: utf-8 -*-

import logging
import os
import time

import numpy as np
import json
import base64
import librosa
import soundfile as sf
import requests
import uuid

import torch

from wetts.vits import utils
from wetts.vits.models import SynthesizerTrn
from wetts.tts_frontend.tts_frontend_model import TTSFrontendModel
from wetts.tts_frontend.tokenizer.prosody_phoneme_model import ProsodyPhonemeModel
from wetts.tts_frontend.tokenizer.prosody_phoneme_processor import ProsodyPhonemeProcessor
from wetts.tts_frontend.tokenizer.vits_tokenizer import VitsTokenizer


logger = logging.getLogger(__name__)


class ModelHandler(object):
    def __init__(self):
        super().__init__()
        self.net_g = None
        self.tts_frontend_model = None
        self.phone_dict = None
        self.context = None
        self.initialized = False
        self.device = None
        self.device = None
        self.org_sr = 22050
        self.tmp_dir = "./tmp"
        os.makedirs(self.tmp_dir, exist_ok=True)

    def initialize(self, context):
        self.context = context
        properties = context.system_properties
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        resource_dir = properties.get("model_dir")

        phone_table_path = os.path.join(resource_dir, "tts_frontend", "phones.txt")
        pretrained_model_name_or_path = os.path.join(resource_dir, "tts_frontend",
                                                     "chinese-electra-180g-small-discriminator")
        prosody_phoneme_pretrained_model_path = \
            os.path.join(resource_dir, "tts_frontend", "prosody_phoneme_chinese-electra-180g-small-discriminator.pt")
        pinyin_lexicon_path = os.path.join(resource_dir, "tts_frontend", "pinyin-lexicon.txt")
        cfg_path = os.path.join(resource_dir, "vits", "base.json")
        checkpoint_path = os.path.join(resource_dir, "vits", "G.pth")

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
        ).to(self.device)

        pp_processor = ProsodyPhonemeProcessor(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            prosody_phoneme_model=pp_model,
            prosody_phoneme_pretrained_model_path=prosody_phoneme_pretrained_model_path,
            device=self.device
        )
        vits_tokenizer = VitsTokenizer(
            pinyin_lexicon_path=pinyin_lexicon_path,
            phone_table_path=phone_table_path
        )
        self.tts_frontend_model = TTSFrontendModel(
            sentence_processor=pp_processor,
            tokenizer=vits_tokenizer,
            device=self.device
        )
        self.initialized = True

    def preprocess(self, requests):
        data_batch = []
        with torch.no_grad():
            for idx, data in enumerate(requests):
                str_dict = {k: v.decode('utf-8') for k, v in data.items()}
                # Convert the string dictionary to a JSON object
                data = json.loads(json.dumps(str_dict))
                speaker_name = data["speaker_name"]
                raw_text = data["text"]
                sample_rate = int(data["sample_rate"])
                if_audio = int(data["if_audio"])
                if_upload = int(data["if_upload"])
                logger.info(f"text: {raw_text}, speaker_name: {speaker_name}, sample_rate:{sample_rate}")
                data_batch.append({"raw_text": raw_text, "speaker_name": speaker_name,
                                   "sample_rate": sample_rate, "if_upload": if_upload, "if_audio": if_audio})
        return data_batch

    def inference(self, data_batch, *args, **kwargs):
        raw_texts = [data["raw_text"] for data in data_batch]
        sample_rates = [data["sample_rate"] for data in data_batch]
        if_uploads = [data["if_upload"] for data in data_batch]
        if_audios = [data["if_audio"] for data in data_batch]
        audios = []
        with torch.no_grad():
            for raw_text in raw_texts:
                tokens = self.tts_frontend_model.parse(raw_text)
                x = tokens.to(self.device).unsqueeze(0)
                x_length = torch.LongTensor([tokens.size(0)]).to(self.device)
                sid = torch.LongTensor([0]).to(self.device)
                audio = self.net_g.infer(
                    x,
                    x_length,
                    sid=sid,
                    noise_scale=.667,
                    noise_scale_w=0.8,
                    length_scale=1)[0][0, 0].data.cpu().float().numpy()
                audio *= 32767 / max(0.01, np.max(np.abs(audio))) * 0.6
                audio = np.clip(audio, -32767.0, 32767.0)
                audios.append(audio)

        output = {
            "audios": audios,
            "sample_rates": sample_rates,
            "if_uploads": if_uploads,
            "if_audios": if_audios
        }
        return output

    def postprocess(self, data_batch):
        audios = data_batch["audios"]
        sample_rates = data_batch["sample_rates"]
        if_uploads = data_batch["if_uploads"]
        if_audios = data_batch["if_audios"]
        new_data_batch = []
        for audio, sample_rate, if_upload, if_audio in zip(audios, sample_rates, if_uploads, if_audios):
            response_data = {"msg": "", "code": "OK", "data": {}, "time": int(time.time())}
            if sample_rate != self.org_sr:
                audio = librosa.resample(audio, self.org_sr, sample_rate)
            if if_audio:
                pcm = audio.astype(np.int16)
                pcm = base64.b64encode(pcm.tobytes()).decode("utf-8")
                response_data["data"]["audio"] = pcm
            if if_upload:
                file_path = os.path.join(self.tmp_dir, f"{str(uuid.uuid4())}.wav")
                try:
                    sf.write(file_path, audio / 36767, samplerate=sample_rate)
                    res = self.upload_file(5, file_path)
                    audio_url = res["url"]
                    response_data["data"]["audio_url"] = audio_url
                except Exception as e:
                    response_data["msg"] = str(e)
                    response_data["code"] = "ERROR"
                    logger.error(f"{str(e)}")
                finally:
                    if os.path.exists(file_path):
                        os.remove(file_path)
            new_data_batch.append(json.dumps(response_data))

        return new_data_batch

    def upload_file(self, upload_type, filename):
        VISION_URL = "http://vision.cc.163.com/inner/upload/token/"
        # 请自行处理http请求异常
        res = requests.get(VISION_URL, params={"type": upload_type, "ttl": 60 * 60 * 24 * 7, "usage": "tts"})
        res.raise_for_status()
        data = res.json()
        url = data.get("url", "")
        headers = data.get("headers", None)
        with open(filename, 'rb') as f:
            data = f.read()
            res = requests.post(url, data=data, headers=headers)
            res.raise_for_status()
            return res.json()

    def handle(self, data, context):
        if not self.initialized:
            self.initialize(context)

        if data is None:
            return None

        data = self.preprocess(data)
        data = self.inference(data)
        data = self.postprocess(data)

        return data
