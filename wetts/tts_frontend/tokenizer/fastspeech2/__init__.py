# -*- coding: utf-8 -*-

import re

from wetts.tts_frontend.tokenizer.fastspeech2 import cleaners


class Text2Sequence:
    def __init__(self, symbols, is_unique=False):
        # Mappings from symbol to numeric ID and vice versa:
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.id_to_symbol = {i: s for i, s in enumerate(symbols)}
        # Regular expression matching text enclosed in curly braces:
        self.curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")
        self.text_cleaners = []
        self.is_unique = is_unique

    def __call__(self, text):
        """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

        The text can optionally have ARPAbet sequences enclosed in curly braces embedded
        in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

        Args:
          text: string to convert to a sequence

        Returns:
          List of integers corresponding to the symbols in the text
        """
        sequence = []

        # Check for curly braces and treat their contents as ARPAbet:
        while len(text):
            m = self.curly_re.match(text)
            if not m:
                sequence += self.symbols_to_sequence(self.clean_text(text, self.text_cleaners))
                break
            sequence += self.symbols_to_sequence(self.clean_text(m.group(1), self.text_cleaners))
            sequence += self.arpabet_to_sequence(m.group(2))
            text = m.group(3)

        return sequence

    def sequence_to_text(self, sequence):
        """Converts a sequence of IDs back to a string"""
        result = ""
        for symbol_id in sequence:
            if symbol_id in self.id_to_symbol:
                s = self.id_to_symbol[symbol_id]
                # Enclose ARPAbet back in curly braces:
                if len(s) > 1 and s[0] == "@":
                    s = "{%s}" % s[1:]
                result += s
        return result.replace("}{", " ")

    def clean_text(self, text, cleaner_names):
        for name in cleaner_names:
            cleaner = getattr(cleaners, name)
            if not cleaner:
                raise Exception("Unknown cleaner: %s" % name)
            text = cleaner(text)
        return text

    def symbols_to_sequence(self, symbols):
        return [self.symbol_to_id[s] for s in symbols if self.should_keep_symbol(s)]

    def arpabet_to_sequence(self, text):
        if self.is_unique:
            return self.symbols_to_sequence(["@" + s for s in text.split()])
        else:
            return self.symbols_to_sequence(text.split())

    def should_keep_symbol(self, s):
        return s in self.symbol_to_id and s != "_" and s != "~"
