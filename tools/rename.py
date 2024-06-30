# -*- coding: utf-8 -*-

import os
import glob


def run():
    root_dir = "/data1/xiepengyuan/audio/tts/jianbaotangzhu-chenwen2/Wave"
    audio_paths = glob.glob(os.path.join(root_dir, "*.wav"))
    for audio_path in audio_paths:
        filename = os.path.basename(audio_path)
        new_filename = filename[:-4]
        os.rename(audio_path, os.path.join(root_dir, new_filename))


if __name__ == '__main__':
    run()
