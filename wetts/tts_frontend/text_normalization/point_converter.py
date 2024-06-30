import re

from wetts.tts_frontend.text_normalization.num_converter import num_converter


class PointConverter(object):
    def __init__(self):
        super().__init__()
        self.pattern = re.compile(r"\d+(?:\.\d+){3,}")

    def convert(self, text):
        chars = list(text)
        match_objs = list(self.pattern.finditer(text))
        for matched_obj in reversed(match_objs):
            matched_text = matched_obj[0]
            numbers = re.split(r"\.", matched_text)
            result = []
            for i, num in enumerate(numbers):
                if i != 0:
                    result.append("点")
                result.extend(num_converter.digital_reading(num, yao=True))
            chars[matched_obj.start(0):matched_obj.end(0)] = result
        text = "".join(chars)
        return text
