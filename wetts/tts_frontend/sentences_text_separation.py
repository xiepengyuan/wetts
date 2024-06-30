# -*- coding: utf-8 -*-

import re


class SentencesTextSeparation:
    def __init__(self):
        self.sentence_boundary_pattern = re.compile(r'((?:[。；;！!？?…]|\.{3,}|\.{2,}\s)+[”’』」）］〕】》〉)\]}]*)')
        self.whitespace_or_punctuation_pattern = re.compile(r'[\s\W]+')
        self.basic_boundary_pattern = re.compile(r'([A-Za-z]+|[0-9]+|.)')
        self.second_boundary_pattern = re.compile(r'([，,：:]+[”’』」）］〕】》〉)\]}]*)')
        self.third_boundary_pattern = re.compile(r'([^\w“‘『「（［〔【《〈(\[{]+)')

    def separate(self, text, boundary_pattern=None):
        """分句"""
        boundary_pattern = boundary_pattern or self.sentence_boundary_pattern
        sentences = []
        for token in boundary_pattern.split(text):
            if token:
                if (boundary_pattern.fullmatch(token) or
                    self.whitespace_or_punctuation_pattern.fullmatch(token)) \
                        and sentences:
                    sentences[-1] += token
                else:
                    sentences.append(token)
        return sentences

    def separate_with_limited_length(self, text, max_len=50, force=True):
        """按照长度尽量切割字符串"""
        ss1 = self.separate(text)
        ss1_tokens = [(s1, self.len_of(s1)) for s1 in ss1]
        ss1_tokens = self.concat_tokens(ss1_tokens, max_len=max_len)
        for s1, s1_len in ss1_tokens:
            if s1_len <= max_len:
                yield s1
            else:
                ss2 = self.separate(s1, self.second_boundary_pattern)
                ss2_tokens = [(s2, self.len_of(s2)) for s2 in ss2]
                ss2_tokens = self.concat_tokens(ss2_tokens, max_len=max_len)
                for s2, s2_len in ss2_tokens:
                    if (not force) or (s2_len <= max_len):
                        yield s2
                    else:
                        ss3 = self.separate(s2, self.third_boundary_pattern)
                        ss3_tokens = [(s3, self.len_of(s3)) for s3 in ss3]
                        ss3_tokens = self.concat_tokens(ss3_tokens, max_len=max_len)
                        for s3, s3_len in ss3_tokens:
                            if s3_len <= max_len:
                                yield s3
                            else:
                                ss4 = self.separate(s3, self.basic_boundary_pattern)
                                ss4_tokens = [(s4, self.len_of(s4)) for s4 in ss4]
                                ss4_tokens = self.concat_tokens(ss4_tokens, max_len=max_len)
                                for s4, s4_len in ss4_tokens:
                                    yield s4

    def len_of(self, text, en_word_len=2.5, others_len=None):
        """专门设计的长度，例如一个英文单词相当于2.5的长度"""
        tokens = self.separate(text, self.basic_boundary_pattern)
        tokens_len = 0
        for token in tokens:
            if len(token) == 1:
                tokens_len += 1
            elif re.fullmatch(r'[A-Za-z]+', token):
                tokens_len += en_word_len
            else:
                if others_len is not None:
                    tokens_len += others_len
                else:
                    tokens_len += len(token)
        return tokens_len

    @staticmethod
    def concat_tokens(tokens, *, max_len=50):
        """拼接较短的token"""
        sum_len = sum([token_len for _, token_len in tokens])
        avg_len = sum_len // (sum_len // max_len + 1)
        new_tokens = []
        for token, token_len in tokens:
            if (not new_tokens) or (new_tokens and new_tokens[-1][1] >= avg_len):
                new_tokens.append((token, token_len))
            elif new_tokens[-1][1] + token_len <= max_len:
                new_tokens[-1] = (new_tokens[-1][0] + token, new_tokens[-1][1] + token_len)
            else:
                new_tokens.append((token, token_len))
        return new_tokens
