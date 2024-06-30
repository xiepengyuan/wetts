# -*- coding: utf-8 -*-
import re
from transformers import AutoTokenizer

import torch
import torch.nn as nn

from wetts.tts_frontend.tokenizer.utils import (
    get_tokenized_inputs, set_prosody, single_pinyin, phrase_pinyin, change_prosody_by_punct, change_phoneme_by_prosody
)
from wetts.tts_frontend.tokenizer.sandhi import fix_yi_bu, fix_last_yi_bu, fix_33


class ProsodyPhonemeProcessor(nn.Module):
    def __init__(
            self,
            pretrained_model_name_or_path,
            prosody_phoneme_model,
            prosody_phoneme_pretrained_model_path=None,
            device="cpu",
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.idx2tag = {0: "O", 1: "B", 2: "I", 3: "E", 4: "S"}

        self.prosody_phoneme_model = prosody_phoneme_model.eval()
        if prosody_phoneme_pretrained_model_path:
            self.prosody_phoneme_model.load_state_dict(
                torch.load(prosody_phoneme_pretrained_model_path, map_location=torch.device(device))
            )
        self.polyphones = list(self.prosody_phoneme_model.counter)
        self.char2phonemes = {char: list(
            self.prosody_phoneme_model.counter[char]) for char in self.prosody_phoneme_model.counter}

        # if self.en_process:
        #     self.g2p_en = G2p()
        self.alpha_pinyin = True

    def __call__(self, sentence, device):
        chars = sentence.chars

        # single
        single_pinyin(sentence, alpha_pinyin=self.alpha_pinyin)

        # phrase and fix_last_yi_bu
        phrase_pinyin(sentence)
        fix_last_yi_bu(sentence)

        # model
        char_texts = sentence.get_chars()
        predict_prosody = self.get_pred_prosody_flag(chars)
        predict_phoneme = self.get_pred_phoneme_flag(chars)

        seq_embed = None
        if predict_prosody or predict_phoneme:
            # use model
            pre_tokenized_inputs = get_tokenized_inputs(char_texts, set(self.tokenizer.vocab))

            encoded = self.tokenizer.batch_encode_plus(
                [pre_tokenized_inputs],
                add_special_tokens=True,
                padding=True,
                is_split_into_words=True,
                return_tensors="pt",
            )
            token_ids = encoded["input_ids"].to(device)
            token_type_ids = encoded["token_type_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            with torch.no_grad():
                seq_embed = self.prosody_phoneme_model.seq_embed(
                    input_ids=token_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                )

        # when use pp_model, if predict_phoneme is True then predict_prosody is True.
        if predict_phoneme:
            predict_prosody = True
        if not predict_prosody:
            # use word info
            for word in sentence.words:
                if word.lang in {"zh", "en"}:
                    word.chars[-1].prosody = "#1"
        else:
            pred_p1_tag_ids, pred_p2_tag_ids, pred_p3_tag_ids = self.prosody_phoneme_model.seq_decode(
                seq_embed, attention_mask=attention_mask)
            sentence = set_prosody(sentence, pred_p1_tag_ids, pred_p2_tag_ids, pred_p3_tag_ids, self.idx2tag)

        change_prosody_by_punct(sentence)
        change_phoneme_by_prosody(sentence)

        if predict_phoneme:
            with torch.no_grad():
                for idx, char in enumerate(chars):
                    if not char.phoneme_changeable:
                        continue
                    char_text = char.text
                    if char_text in self.prosody_phoneme_model.counter:
                        position_ids = torch.LongTensor([idx + 1]).to(device)
                        polyphone_ids = torch.LongTensor(
                            [self.polyphones.index(char_text)]).to(device)
                        phoneme_sizes = torch.LongTensor(
                            [len(self.char2phonemes[char_text])]).to(device)
                        output = self.prosody_phoneme_model.log_softmax_out(
                            seq_embed, position_ids, polyphone_ids, phoneme_sizes)
                        pred_phoneme_ids = torch.argmax(output, dim=-1)
                        pred_phoneme_id = pred_phoneme_ids[0]
                        phoneme = self.char2phonemes[char_text][pred_phoneme_id]
                        char.phoneme = phoneme
                        char.phoneme_src = "model"

        # sandhi
        fix_yi_bu(sentence)
        fix_33(sentence)

        # english
        # if self.en_process:
        #     sentence = self.g2p_en.g2p(sentence)

        # 取消英文字母的韵律，tokenizer再添加
        for char in chars:
            if char.lang == "en":
                char.prosody = None

        return sentence

    def get_pred_phoneme_flag(self, chars):
        predict_phoneme = False
        for char in chars:
            if char.text in self.char2phonemes:
                if char.phoneme_changeable:
                    predict_phoneme = True
                    break
        return predict_phoneme

    def get_pred_prosody_flag(self, chars):
        length = 5
        if len(chars) <= length:
            return False
        predict_prosody = False
        chunk_len = 0
        for char in chars:
            if char.lang == "zh":
                chunk_len += 1
            else:
                chunk_len = 0
            if chunk_len > length:
                predict_prosody = True
                break
        return predict_prosody

    @staticmethod
    def sentences_to_text(sentences):
        pinyins = []
        for sentence in sentences:
            phoneme_prosody = sentence.get_phoneme(with_prosody=True)
            pys = re.split(r"\s+", phoneme_prosody)
            pys = [py for py in pys if re.fullmatch(r"#\d|[a-z]+\d|[A-Z]|[、：；，。？！—…]", py)]
            pinyins.extend(pys)

        while len(pinyins) > 0 and re.fullmatch(r"#\d|\W", pinyins[0]):
            pinyins = pinyins[1:]
        py_idx = len(pinyins) - 1
        while py_idx >= 0 and re.fullmatch(r"#\d|\W", pinyins[py_idx]):
            py_idx -= 1
        for idx, pinyin in enumerate(pinyins):
            if idx >= py_idx:
                break
            if re.fullmatch(r"#4", pinyin):
                pinyins[idx] = "#4 #4"
        text = " ".join(pinyins)
        return text


def test_prosody_phoneme_processor():
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    cfg = OmegaConf.load("/data1/xiepengyuan/workspace/NeMo/nemo/collections/nlp/modules/tts_frontend/tokenizer/tokenizer.yaml")
    prosody_phoneme_processor = instantiate(cfg.prosody_phoneme_processor)


def test():
    from wetts.tts_frontend.tokenizer.prosody_phoneme_model import ProsodyPhonemeModel
    pretrained_model_name_or_path = "/data1/xiepengyuan/workspace/audio/wetts/resources/tts_frontend/chinese-electra-180g-small-discriminator"
    prosody_phoneme_pretrained_model_path = "/data1/xiepengyuan/workspace/audio/wetts/resources/tts_frontend/prosody_phoneme_chinese-electra-180g-small-discriminator.pt"
    pp_model = ProsodyPhonemeModel(
        tagset_size=5,
        hidden_dim=128,
        pretrained_model_name_or_path=pretrained_model_name_or_path
    )
    pp_processor = ProsodyPhonemeProcessor(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        prosody_phoneme_model=pp_model,
        prosody_phoneme_pretrained_model_path=prosody_phoneme_pretrained_model_path
    )


if __name__ == "__main__":
    test()
