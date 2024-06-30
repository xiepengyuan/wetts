# -*- coding: utf-8 -*-

def gen():
    with open("/data1/xiepengyuan/workspace/audio/wetts/resources/tts_frontend/pinyin-lexicon.txt", "w") as f:
        with open("/data1/xiepengyuan/workspace/audio/wetts/resources/tts_frontend/pinyin-lexicon-rr.txt", "r") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.strip()
                line_split = line.split(" ")
                if line_split[1] in ["w", "y"]:
                    new_str = line_split[0] + " " + " ".join(line_split[2:])
                else:
                    new_str = line
                f.write(f"{new_str}\n")


if __name__ == '__main__':
    gen()
