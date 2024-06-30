import sys
import os
import shutil

template_dir = "/data1/xiepengyuan/workspace/audio/wetts/examples/ljspeech-mix_v2/data/ljspeech-mix_v2"
zh_train_path = os.path.join(template_dir, "zh_train.txt")
zh_val_path = os.path.join(template_dir, "zh_val.txt")
en_train_path = os.path.join(template_dir, "en_train.txt")
en_val_path = os.path.join(template_dir, "en_val.txt")
phone_table_path = os.path.join(template_dir, "phones.txt")

raw_data_dir = sys.argv[1]
data_dir = sys.argv[2]
os.makedirs(data_dir, exist_ok=True)

dst_train_path = os.path.join(data_dir, "train.txt")
dst_val_path = os.path.join(data_dir, "val.txt")
dst_phone_table_path = os.path.join(data_dir, "phones.txt")

shutil.copy(phone_table_path, dst_phone_table_path)


# train
train_lines = []
with open(zh_train_path) as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        filename, text = line.split("|")
        path = os.path.join(raw_data_dir, "zh_wave", filename)
        line = f"{path}|{text}"
        train_lines.append(line)
with open(en_train_path) as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        filename, text = line.split("|")
        path = os.path.join(raw_data_dir, "en_wave", filename)
        line = f"{path}|{text}"
        train_lines.append(line)
with open(dst_train_path, "w") as f:
    for train_line in train_lines:
        f.write(f"{train_line}\n")

# val
val_lines = []
with open(zh_val_path) as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        filename, text = line.split("|")
        path = os.path.join(raw_data_dir, "zh_wave", filename)
        line = f"{path}|{text}"
        val_lines.append(line)
with open(en_val_path) as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        filename, text = line.split("|")
        path = os.path.join(raw_data_dir, "en_wave", filename)
        line = f"{path}|{text}"
        val_lines.append(line)
with open(dst_val_path, "w") as f:
    for val_line in val_lines:
        f.write(f"{val_line}\n")

