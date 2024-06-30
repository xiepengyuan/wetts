import re
import jieba.posseg as pseg

from wetts.tts_frontend.pinyin2ipa import pinyin_converter
from wetts.tts_frontend.arpabet_convertor import arpa_to_ipa


RE_HANS = re.compile(
    r'^(?:['
    r'\u3007'  # 〇
    r'\u3400-\u4dbf'  # CJK扩展A:[3400-4DBF]
    r'\u4e00-\u9fff'  # CJK基本:[4E00-9FFF]
    r'\uf900-\ufaff'  # CJK兼容:[F900-FAFF]
    r'\U00020000-\U0002A6DF'  # CJK扩展B:[20000-2A6DF]
    r'\U0002A703-\U0002B73F'  # CJK扩展C:[2A700-2B73F]
    r'\U0002B740-\U0002B81D'  # CJK扩展D:[2B740-2B81D]
    r'\U0002F80A-\U0002FA1F'  # CJK兼容扩展:[2F800-2FA1F]
    r'])+$'
)


def get_language(text):
    if RE_HANS.fullmatch(text):
        return "zh"
    elif re.fullmatch(r"\W+", text):
        return "punct"
    elif re.fullmatch(r"[a-zA-Z]+[-']*[a-zA-Z]*", text):
        return "en"
    else:
        return "others"


class Sentence(object):
    def __init__(self, text):
        self.text = text
        self.words = self.set_words(text)
        self.chars = [char for word in self.words for char in word.chars]

    def get_word_pos(self):
        return " ".join([f"{word.text}/{word.pos}" for word in self.words])

    @staticmethod
    def set_words(text):
        word_flags = pseg.lcut(text)
        words = []
        for idx, (word_text, word_pos) in enumerate(word_flags):
            lang = get_language(word_text)
            if lang == "en":
                for char in word_text:
                    words.append(Word(char, pos=word_pos, lang=lang))
                continue
            if (idx == 0) or (idx == len(word_flags) - 1):
                words.append(Word(word_text, pos=word_pos, lang=lang))
            elif re.fullmatch(r"\S+", word_text):
                words.append(Word(word_text, pos=word_pos, lang=lang))
            elif get_language(word_flags[idx - 1].word) == "en":
                continue
            elif get_language(word_flags[idx + 1].word) == "en":
                continue
            else:
                words.append(Word(word_text, pos=word_pos, lang=lang))
        return words

    def get_chars(self):
        return [char.text for char in self.chars]

    def get_prosody_text(self):
        tokens = []
        for idx, char in enumerate(self.chars):
            if idx > 0 and char.lang == "en" and self.chars[idx - 1].lang == "en":
                tokens.append(" ")
            tokens.append(char.text)
            if char.prosody is not None:
                tokens.append(char.prosody)
        return "".join(tokens)

    def get_phoneme(self, with_prosody=False, with_src=False, to_ipa=False):
        tokens = []
        for char in self.chars:
            if re.fullmatch(r"\s+", char.text):
                continue
            if char.phoneme is not None:
                phoneme = char.phoneme

                if to_ipa:
                    if char.lang == "zh":
                        phoneme = pinyin_converter.convert_pinyin(phoneme) + phoneme[-1]
                    elif char.lang == "en":
                        phoneme = arpa_to_ipa(phoneme)

                if with_src and char.phoneme_src:
                    if char.phoneme_src == "phrase":
                        tokens.append(f"({phoneme})")
                    elif char.phoneme_src == "model":
                        tokens.append(f"{{{phoneme}}}")
                    else:
                        tokens.append(phoneme)
                else:
                    tokens.append(phoneme)
            else:
                tokens.append(char.text)

            if with_prosody:
                if char.prosody is not None:
                    tokens.append(char.prosody)

        return " ".join(tokens)


class Word(object):
    def __init__(self, text, pos=None, lang=None):
        self.text = text
        self.pos = pos  # Part of speech
        self.lang = lang
        self.chars = self.get_chars(text)

    def get_chars(self, text):
        if self.lang is None:
            self.lang = get_language(text)
        if self.lang == "zh":
            chars = list(text)
            return [Char(char, self.lang) for char in chars]
        else:
            return [Char(text, self.lang)]


class Char(object):
    def __init__(self, text, lang, prosody=None, phoneme=None):
        self.text = text
        self.lang = lang
        self.prosody = prosody
        self.phoneme = phoneme

        self.prosody_changeable = self.get_changeable(prosody)
        self.phoneme_changeable = self.get_changeable(phoneme)
        self.phoneme_src = None

    @staticmethod
    def get_changeable(x):
        if x is None:
            return True
        else:
            return False


def main():
    texts = [
        "Hello，12块钱，今天天气很晴朗，处处好风光。",
        "拥有行业领先的语音合成全链路技术，合成速度快、准确率高。",
    ]
    for text in texts:
        print("\nraw text:", text)
        sentence = Sentence(text)

        word_pos = sentence.get_word_pos()
        print("word_pos:", word_pos)

        chars = sentence.get_chars()
        print("chars:", chars)


if __name__ == "__main__":
    main()
