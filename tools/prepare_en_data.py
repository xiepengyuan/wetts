# -*- coding: utf-8 -*-

def run():
    train_data_path = "/data1/xiepengyuan/workspace/audio/wetts/examples/ljspeech-mix_v2/data/ljspeech-mix_v2/train.txt"
    val_data_path = "/data1/xiepengyuan/workspace/audio/wetts/examples/ljspeech-mix_v2/data/ljspeech-mix_v2/val.txt"
    data_list_path = "/data1/xiepengyuan/workspace/audio/wetts/examples/ljspeech-mix_v2/data/ljspeech-mix_v2/data.list"

    filenames = []
    for data_path in [train_data_path, val_data_path]:
        with open(data_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                file_path, _ = line.split("|")
                filenames.append(file_path.split("/")[-1])
    filenames.sort()
    with open(data_list_path, "w") as f:
        for filename in filenames:
            f.write(filename + "\n")

    train_data_path = "/data1/xiepengyuan/workspace/audio/wetts/examples/ljspeech-mix_v2/data/ljspeech-mix_v2/ljs_audio_text_train_filelist.txt"
    val_data_path = "/data1/xiepengyuan/workspace/audio/wetts/examples/ljspeech-mix_v2/data/ljspeech-mix_v2/ljs_audio_text_test_filelist.txt"
    data_list_path = "/data1/xiepengyuan/workspace/audio/wetts/examples/ljspeech-mix_v2/data/ljspeech-mix_v2/data.txt"

    datas = []
    for data_path in [train_data_path, val_data_path]:
        with open(data_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                file_path, _ = line.split("|")
                filename = file_path.split("/")[-1]
                if filename in filenames:
                    datas.append(line)
    datas.sort()
    with open(data_list_path, "w") as f:
        for data in datas:
            f.write(data + "\n")


def run2():
    src_data_path = "/data1/xiepengyuan/workspace/audio/wetts/examples/ljspeech-mix_v2/data/ljspeech-mix_v2/ljs_audio_text_train_filelist.txt.cleaned"
    dst_data_path = "/data1/xiepengyuan/workspace/audio/wetts/examples/ljspeech-mix_v2/data/ljspeech-mix_v2/010001-020000.txt.cleaned"
    data_map = "/data1/xiepengyuan/workspace/audio/wetts/examples/ljspeech-mix_v2/data/ljspeech-mix_v2/data_map.txt.cleaned"

    index = 10000
    max_index = 20000
    with open(data_map, "w") as f_map:
        with open(dst_data_path, "w") as f_dst:
            with open(src_data_path, "r") as f_src:
                lines = f_src.readlines()
                for line in lines:
                    line = line.strip()
                    file_path, text = line.split("|")
                    filename = file_path.split("/")[-1]
                    index += 1
                    if index > max_index:
                        break
                    new_filename = f"{str(index).zfill(6)}.wav"
                    f_map.write(f"{new_filename} {filename}\n")
                    f_dst.write(f"{new_filename}|{text}\n")


if __name__ == '__main__':
    run2()
