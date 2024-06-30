# -*- coding: utf-8 -*-

import re

import torch

from wetts.tts_frontend.utils import general_padding
from wetts.tts_frontend.text_normalization.text_normalization import TextNormalization
from wetts.tts_frontend.sentences_text_separation import SentencesTextSeparation
from wetts.tts_frontend.sentence import Sentence


class TTSFrontendModel:
    def __init__(self, sentence_processor, tokenizer, device="cpu"):
        self.text_normalization = TextNormalization()
        self.sentences_text_separation = SentencesTextSeparation()
        self.sentence_processor = sentence_processor
        self.tokenizer = tokenizer
        self.device = device

    def parse(self, text):
        text = self.text_normalization(text)
        sentence_texts = self.sentences_text_separation.separate(text)
        sentences = []
        for sentence_text in sentence_texts:
            sentence = Sentence(sentence_text)
            sentence = self.sentence_processor(sentence, self.device)
            sentences.append(sentence)
        text = self.sentence_processor.sentences_to_text(sentences)
        print(text)
        tokens = self.tokenizer(text)
        tokens = torch.tensor(tokens, dtype=torch.long).to(self.device)
        return tokens

    def to_phoneme_text(self, text):
        text = self.text_normalization(text)
        sentence_texts = self.sentences_text_separation.separate(text)
        sentences = []
        for sentence_text in sentence_texts:
            sentence = Sentence(sentence_text)
            sentence = self.sentence_processor(sentence, self.device)
            sentences.append(sentence)
        text = self.sentence_processor.sentences_to_text(sentences)
        phoneme_text = self.tokenizer.to_phoneme_text(text)
        return phoneme_text

    def parse_batch(self, texts):
        tokens_batch = []
        for text in texts:
            tokens = self.parse(text)
            tokens_batch.append(tokens)
        padded_tokens_batch = []
        tokens_lens = []
        for tokens in tokens_batch:
            tokens_len = torch.tensor(tokens.shape[-1], dtype=torch.long).to(tokens.device)
            tokens_lens.append(tokens_len)
        max_tokens_len = max(tokens_lens).item()
        for tokens, tokens_len in zip(tokens_batch, tokens_lens):
            padded_tokens = general_padding(tokens, tokens_len.item(), max_tokens_len, pad_value=0)
            padded_tokens_batch.append(padded_tokens)
        padded_tokens_batch = torch.stack(padded_tokens_batch)
        tokens_lens = torch.stack(tokens_lens)
        return padded_tokens_batch, tokens_lens

    def to_pinyin(self, text):
        text = self.text_normalization(text)
        sentence_texts = self.sentences_text_separation.separate(text)
        sentences = []
        for sentence_text in sentence_texts:
            sentence = Sentence(sentence_text)
            sentence = self.sentence_processor(sentence, self.device)
            sentences.append(sentence)
        text = self.sentence_processor.sentences_to_text(sentences)
        tokens = re.split(r"\s+", text)
        tokens = [token for token in tokens if token]
        pinyin = []
        for token in tokens:
            if re.fullmatch(r"#\d|\W", token):
                continue
            else:  # pinyin
                pinyin.append(token)
        pinyin = " ".join(pinyin)
        return pinyin


def test():
    from wetts.tts_frontend.tokenizer.prosody_phoneme_model import ProsodyPhonemeModel
    from wetts.tts_frontend.tokenizer.prosody_phoneme_processor import ProsodyPhonemeProcessor
    from wetts.tts_frontend.tokenizer.vits_tokenizer import VitsTokenizer
    import os
    resource_dir = "/data1/xiepengyuan/workspace/audio/wetts/resources/v1"
    pretrained_model_name_or_path = os.path.join(resource_dir, "tts_frontend",
                                                 "chinese-electra-180g-small-discriminator")
    prosody_phoneme_pretrained_model_path = \
        os.path.join(resource_dir, "tts_frontend", "prosody_phoneme_chinese-electra-180g-small-discriminator.pt")
    pinyin_lexicon_path = os.path.join(resource_dir, "tts_frontend", "pinyin-lexicon.txt")
    phone_table_path = os.path.join(resource_dir, "tts_frontend", "phones.txt")
    device = "cuda"

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
    tts_frontend_model = TTSFrontendModel(
        sentence_processor=pp_processor,
        tokenizer=vits_tokenizer,
        device=device
    )
    tokens = tts_frontend_model.parse("而Treyarch hello工作室的系列新作则会向后延期")
    print(tokens)


if __name__ == '__main__':
    test()
