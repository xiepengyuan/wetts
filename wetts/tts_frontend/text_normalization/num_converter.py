# -*- coding: utf-8 -*-

import re


class NumConverter(object):
    def __init__(self):
        super().__init__()
        self.num2char = {num: char for num, char in zip("0123456789", "零一二三四五六七八九")}
        self.num2char_yao = {num: char for num, char in zip("0123456789", "零幺二三四五六七八九")}
        self.liang = "两"
        self.small_num_unit = "十百千"
        self.big_num_unit = "万亿兆京垓秭穰沟涧正载"
        self.small_unit_len = [
            (unit, i + 1) for i, unit in enumerate(self.small_num_unit)
        ]
        self.big_num_len = [
            (unit, (i + 1) * 4) for i, unit in enumerate(self.big_num_unit)
        ]
        self.unit_len = self.small_unit_len + self.big_num_len
        self.unit2len = dict(self.unit_len)
        self.point = "点"
        self.fraction_describe = "分之"
        self.baiqianwan_fenhao = ["百分之", "千分之", "万分之"]
        self.sign2char = {sign: char for sign, char in zip("+-", "正负")}
        self.decimal_pattern = "[+-]*(?:\d+\.?\d*)"

    def convert(self, text, is_value=False):
        if is_value:
            text = re.sub(
                rf"{self.decimal_pattern}(?:/{self.decimal_pattern})?[%％‰‱]?",
                lambda x: "".join(self.signed_number(x[0])),
                text,
            )
        else:
            text = re.sub(
                r"\d+", lambda x: "".join(self.digital_reading(x[0])), text
            )
        return text

    def digital_reading(self, text, yao=False):
        if yao:
            return [self.num2char_yao[num] for num in text]
        else:
            return [self.num2char[num] for num in text]

    def value_reading(self, text, liang=True):
        text_lstrip0 = text.lstrip("0")
        if len(text_lstrip0) == 0:
            return [self.num2char["0"]]
        else:
            value = self._value_reading(text_lstrip0)

            # liang
            for i, v in enumerate(value):
                if v == self.num2char["2"]:
                    pre_v = value[i - 1] if i > 0 else None
                    next_v = value[i + 1] if i + 1 < len(value) else None
                    if not (
                            next_v is None
                            or next_v == self.small_num_unit[0]
                            or (pre_v is not None and pre_v == self.small_num_unit[0])
                            or (pre_v is not None and next_v in self.big_num_unit)
                    ):
                        value[i] = self.liang
            # single 2
            if liang:
                if len(value) == 1 and value[0] == self.num2char["2"]:
                    pass
                    # value[0] = self.liang

            # 10, 11, 12, ..., 19
            if (
                    len(value) >= 2
                    and value[0] == self.num2char["1"]
                    and value[1] == self.small_num_unit[0]
            ):
                value = value[1:]

            return value

    def _value_reading(self, text):
        text_lstrip0 = text.lstrip("0")
        if len(text_lstrip0) == 0:
            return []
        elif len(text_lstrip0) == 1:
            if len(text) != len(text_lstrip0):
                return [self.num2char["0"], self.num2char[text_lstrip0]]
            else:
                return [self.num2char[text_lstrip0]]
        elif len(text_lstrip0) > 48:
            return self.digital_reading(text)
        else:
            unit, length = next(
                (u, l) for u, l in reversed(self.unit_len) if l < len(text_lstrip0)
            )
            text_l, text_r = text[:-length], text_lstrip0[-length:]
            return self._value_reading(text_l) + [unit] + self._value_reading(text_r)

    def decimal(self, text):
        parts = re.split(r"\.", text, maxsplit=1)
        if len(parts) == 1:
            int_part, dec_part = parts[0], ""
        else:
            int_part, dec_part = parts[0], parts[1]
        if int_part:
            int_value = self.value_reading(int_part)
        else:
            int_value = [self.num2char["0"]]
        if dec_part:
            dec_value = self.digital_reading(dec_part)
            return int_value + [self.point] + dec_value
        else:
            return int_value

    def fraction(self, text):
        parts = re.split(r"/", text, maxsplit=1)
        if len(parts) == 1:
            return self.decimal(text)
        else:
            numerator, denominator = parts[0], parts[1]
            return (
                    self.decimal(denominator)
                    + [self.fraction_describe]
                    + self.decimal(numerator)
            )

    def percentage(self, text):
        if re.search(r"[%％]$", text):
            return [self.baiqianwan_fenhao[0]] + self.fraction(text[:-1])
        elif re.search(r"[‰]$", text):
            return [self.baiqianwan_fenhao[1]] + self.fraction(text[:-1])
        elif re.search(r"[‱]$", text):
            return [self.baiqianwan_fenhao[2]] + self.fraction(text[:-1])
        else:
            return self.fraction(text)

    def signed_number(self, text):
        print("匹配到:", text)
        match_obj = re.match(r"^[+-]+", text)
        if match_obj is not None:
            all_sign = []
            for sign in match_obj[0]:
                all_sign.append(self.sign2char[sign])
            unsigned_text = re.sub(r"^[+-]+", "", text)
            return all_sign + self.percentage(unsigned_text)
        else:
            return self.percentage(text)


num_converter = NumConverter()
