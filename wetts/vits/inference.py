# -*- coding: utf-8 -*-

import os

import numpy as np
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
        tokens = self.tts_frontend_model.parse(text)
        with torch.no_grad():
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
        return audio
