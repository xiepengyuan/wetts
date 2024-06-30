import re

from wetts.tts_frontend.text_normalization.num_converter import num_converter


class PhoneConverter(object):
    def __init__(self):
        super().__init__()
        self.pattern = re.compile(r"(?<!\d)(\+?(\d+)\s*)(1[3-9]\d{9})(?!\d)")
        self.pattern1 = re.compile(r"(?<!\d)(\d{2,4})-(\d{8})(?!\d)")
        self.pattern2 = re.compile(r"(#[*#]*)(\d+)([*#]*)")
        self.special2char = {s: c for s, c in zip("#*", "井星")}

    def convert(self, text):
        chars = list(text)
        match_objs = list(self.pattern.finditer(text))
        for i in reversed(match_objs):
            phone1, phone2 = i[2], i[3]
            chars[i.start(3):i.end(3)] = num_converter.digital_reading(phone2, yao=True)
            chars[i.start(1):i.end(1)] = num_converter.digital_reading(phone1, yao=True)
        text = "".join(chars)

        chars = list(text)
        match_objs = list(self.pattern1.finditer(text))
        for i in reversed(match_objs):
            phone1, phone2 = i[1], i[2]
            chars[i.start(2):i.end(2)] = num_converter.digital_reading(phone2, yao=True)
            chars[i.start(1):i.end(1) + 1] = num_converter.digital_reading(phone1, yao=True)
        text = "".join(chars)

        chars = list(text)
        match_objs = list(self.pattern2.finditer(text))
        for i in reversed(match_objs):
            phone = i[2]
            for ii in range(i.start(3), i.end(3)):
                chars[ii] = self.special2char[chars[ii]]
            for ii in range(i.start(1), i.end(1)):
                chars[ii] = self.special2char[chars[ii]]
            chars[i.start(2):i.end(2)] = num_converter.digital_reading(phone, yao=True)
        text = "".join(chars)

        return text
