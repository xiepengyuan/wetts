# -*- coding: utf-8 -*-

import re


def convert_text(text):
    # 匹配数字和汉字拼音组合的正则表达式
    pattern = re.compile(r'#\d+')

    # 通过正则表达式将数字和汉字拼音分别提取出来，并进行合并
    result = pattern.sub('', text)

    return result


def run():
    with open("./label/baker_text.txt", 'w') as f:
        with open("./label/000001-010000.txt", 'r', encoding='utf8') as fin:
            lines = fin.readlines()
            for i in range(0, len(lines), 2):
                line = lines[i].strip()
                if line == '':
                    continue
                parts = line.split('\t')
                f.write(parts[0] + '\t' + convert_text(parts[1]) + "\n")


if __name__ == '__main__':
    run()
