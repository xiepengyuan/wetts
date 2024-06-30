# -*- coding: utf-8 -*-

import opencc

from wetts.tts_frontend.text_normalization.full_half_conversion import space_letter_digit_full_to_half
from wetts.tts_frontend.text_normalization.defined_tn import defined_convert
from wetts.tts_frontend.text_normalization.num_converter import num_converter
from wetts.tts_frontend.text_normalization.point_converter import PointConverter
from wetts.tts_frontend.text_normalization.date_converter import DateConverter
from wetts.tts_frontend.text_normalization.time_converter import TimeConverter
from wetts.tts_frontend.text_normalization.phone_converter import PhoneConverter
from wetts.tts_frontend.text_normalization.unit_converter import UnitConverter
from wetts.tts_frontend.text_normalization.digit_converter import DigitConverter
from wetts.tts_frontend.text_normalization.value_converter import ValueConverter
from wetts.tts_frontend.text_normalization.score_converter import ScoreConverter
from wetts.tts_frontend.text_normalization.punct_converter import PunctConverter


class TextNormalization(object):
    def __init__(self):
        super().__init__()
        self.t2s_converter = opencc.OpenCC("t2s.json")
        self.point_converter = PointConverter()
        self.date_converter = DateConverter()
        self.time_converter = TimeConverter()
        self.phone_converter = PhoneConverter()
        self.unit_converter = UnitConverter()
        self.digit_converter = DigitConverter()
        self.value_converter = ValueConverter()
        self.score_converter = ScoreConverter()
        self.punct_converter = PunctConverter()

    def __call__(self, text):
        # 全角转半角
        text = space_letter_digit_full_to_half(text)

        # 繁体转简体
        text = self.t2s_converter.convert(text)

        # 自定义转换
        text = defined_convert(text)

        # 标点
        text = self.punct_converter.pre_convert(text)

        # 数字+.
        text = self.point_converter.convert(text)

        # 日期
        text = self.date_converter.convert(text)

        # 时间
        text = self.time_converter.convert(text)

        # 单位
        text = self.unit_converter.convert(text)

        # 电话
        text = self.phone_converter.convert(text)

        # 数码
        text = self.digit_converter.convert(text)

        # 数值
        text = self.value_converter.convert(text)

        # 比分
        text = self.score_converter.convert(text)

        # 标点
        text = self.punct_converter.convert(text)

        # 数值读法
        text = num_converter.convert(text, is_value=True)

        # 英文转大写（暂时只能读大写，且是逐字母读）
        text = text.upper()

        return text
