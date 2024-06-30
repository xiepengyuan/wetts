# -*- coding: utf-8 -*-

import string

from collections import ChainMap

# Some dictionaries for converting half-angle characters to full-angle characters
space_half2full = {chr(0x20): chr(0x3000)}
letter_half2full = {
    letter: chr(ord(letter) + 0xFEE0) for letter in string.ascii_letters
}
digit_half2full = {digit: chr(ord(digit) + 0xFEE0) for digit in string.digits}
punctuation_half2full = {
    punct: chr(ord(punct) + 0xFEE0) for punct in string.punctuation
}

# Dictionaries for conversion of space, letters and numbers
space_letter_digit_half2full = ChainMap(
    space_half2full, letter_half2full, digit_half2full,
)
space_letter_digit_full2half = {v: k for k, v in space_letter_digit_half2full.items()}

# Dictionaries for conversion of space, letters, numbers and punctuation marks
half2full = ChainMap(
    space_half2full, letter_half2full, digit_half2full, punctuation_half2full,
)
full2half = {v: k for k, v in half2full.items()}


def punctuation_half_to_full(text):
    """半角转全角，只转标点符号"""
    return "".join([punctuation_half2full.get(i, i) for i in text])


def space_letter_digit_full_to_half(text):
    """全角转半角，只转空格、字母和数字"""
    return "".join([space_letter_digit_full2half.get(i, i) for i in text])


def half_to_full(text):
    """半角转全角，对于空格、字母、数字和标点符号"""
    return "".join([half2full.get(i, i) for i in text])


def full_to_half(text):
    """全角转半角，对于空格、字母、数字和标点符号"""
    return "".join([full2half.get(i, i) for i in text])
