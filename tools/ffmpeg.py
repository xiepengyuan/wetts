import subprocess
import glob
import os
import math
import numpy as np
import multiprocessing
import shutil
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
import argparse
import time


def preprocessed(root_path, ext="wav", with_speaker_dir=True):
    items = []
    if with_speaker_dir:
        audio_paths = glob.glob(f"{root_path}/*/*.{ext}", recursive=True)
        for audio_path in audio_paths:
            speaker_id = os.path.basename(os.path.dirname(audio_path))
            name = os.path.splitext(os.path.basename(audio_path))[0]
            items.append(["", audio_path, speaker_id, name])
    else:
        audio_paths = glob.glob(f"{root_path}/*.{ext}", recursive=True)
        for audio_path in audio_paths:
            name = os.path.splitext(os.path.basename(audio_path))[0]
            items.append(["", audio_path, "", name])

    return items


def create_commands(src_files, dst_files, opt=None):
    commands = ""
    if opt is None:
        opt = ""
    for src_file, dst_file in zip(src_files, dst_files):
        command = f"ffmpeg -y -loglevel quiet -i \"{src_file}\" {opt} \"{dst_file}\""
        print(command)
        commands += command + "\n"
    return commands


def write_commands(commands_per_process, index, tmp_dir):
    with open(f"{tmp_dir}/ffmpeg_commands_{index}", "w") as f:
        f.write(commands_per_process)
        f.write(f"rm {tmp_dir}/ffmpeg_commands_{index}")


def multiprocess_ffmpeg(audio_paths, speaker_ids, names, dst_dir, opt, num_processors, verbose=True):
    num_paths_pre_proc = math.ceil(len(audio_paths) / num_processors)
    timestamp = str(int(time.time()))
    tmp_dir = f"./.ffmpeg_tmp_{timestamp}"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    # write commands
    if verbose:
        start_time = time.time()
        print(f"num files: {len(audio_paths)}")
    for i in range(num_processors):
        start = num_paths_pre_proc * i
        end = (i + 1) * num_paths_pre_proc if (i + 1) * num_paths_pre_proc < len(audio_paths) else len(audio_paths)
        dst_paths = []
        for audio_path, speaker_id, name in zip(audio_paths[start:end], speaker_ids[start:end], names[start:end]):
            if speaker_id is not None and speaker_id != "":
                dst_paths.append(os.path.join(dst_dir, speaker_id, f"{name}.wav"))
            else:
                dst_paths.append(os.path.join(dst_dir, f"{name}.wav"))
        commands_per_process = create_commands(audio_paths[start:end], dst_paths, opt)
        write_commands(commands_per_process, i, tmp_dir, )
    if verbose:
        start_time = time.time()
        print(f"write commands time: {time.time() - start_time}")

    # exec commands
    processes = []
    for i in range(num_processors):
        process = subprocess.Popen(f"/bin/bash {tmp_dir}/ffmpeg_commands_{i}", shell=True, universal_newlines=True,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append(process)
    for process in processes:
        out, err = process.communicate()
    if verbose:
        print(f"used time: {time.time() - start_time}s")
    shutil.rmtree(tmp_dir)


def run_multiprocess_ffmpeg(cfg: DictConfig, selected_ids=None):
    print(OmegaConf.to_yaml(cfg))
    datasets = cfg.datasets
    opt = cfg.opt
    num_commands = cfg.num_commands
    num_processors = min(multiprocessing.cpu_count(), cfg.max_num_processors)
    print(f"num_processors: {num_processors}")
    start_time = time.time()
    for dataset in datasets:
        dataset_name = dataset["name"]
        src_dir = dataset["src_dir"]
        dst_dir = dataset["preprocessed_dir"]

        print(f"process {dataset_name} {src_dir} {dst_dir}")

        items = preprocessed(src_dir, ext=cfg.ext, with_speaker_dir=dataset.with_speaker_dir)
        if dataset.with_speaker_dir and selected_ids:
            speaker_items_dict = {}
            for item in items:
                speaker_id = item[2]
                if speaker_id in selected_ids:
                    speaker_items_dict.setdefault(speaker_id, []).append(item)
            items = []
            for speaker_id in selected_ids:
                items += speaker_items_dict[speaker_id]
        audio_paths = [item[1] for item in items]
        speaker_ids = [item[2] for item in items]
        names = [item[3] for item in items]

        if dataset.with_speaker_dir:
            for speaker_id in np.unique(speaker_ids):
                save_speaker_dir = os.path.join(dst_dir, speaker_id)
                os.makedirs(save_speaker_dir, exist_ok=True)

        for start in tqdm(range(0, len(audio_paths), num_commands * num_processors)):
            end = start + num_commands * num_processors
            multiprocess_ffmpeg(audio_paths[start:end], speaker_ids[start:end], names[start:end],
                                dst_dir, opt, num_processors)
    print(f"total used time: {time.time() - start_time}")


def run_ffmpeg():
    start_time = time.time()
    opt = "-ar 16000 -ac 1"
    src_dir = "/data1/xiepengyuan/audio/ccspeakers/2021-11-25/test_selected_le60_v6/65005"
    dst_dir = "/data1/xiepengyuan/audio/ccspeakers/2021-11-25/test_selected_le60_v6_wav/65005"
    os.makedirs(dst_dir, exist_ok=True)
    src_files = glob.glob(os.path.join(src_dir, "*", "*"))
    dst_files = []
    for src_file in src_files[:30]:
        name = os.path.splitext(os.path.basename(src_file))[0]
        speaker_id = os.path.basename(os.path.dirname(src_file))
        dst_path = os.path.join(dst_dir, speaker_id, f"{name}.wav")
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        dst_files.append(dst_path)
    commands = create_commands(src_files, dst_files, opt)
    processes = []
    for command in tqdm(commands.split("\n")):
        print(command)
        process = subprocess.Popen(command, shell=True, universal_newlines=True,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append(process)
    for process in processes:
        out, err = process.communicate()
    used_time = time.time() - start_time
    print(f"ffmpeg used time: {used_time}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--selected_ids',
        nargs='+'
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # run_ffmpeg()
    run_multiprocess_ffmpeg(OmegaConf.load("/data1/xiepengyuan/workspace/audio/wetts/examples/tongtong/tongtong_ffmpeg_config.yaml"))
    # run_multiprocess_ffmpeg(OmegaConf.load("F:\\workspace\\audio\\wetts\\examples\\jianbaotangzhu\\jianbaotangzhu_ffmpeg_local_config.yaml"))
    # ARG = get_args()
    # run_multiprocess_ffmpeg(OmegaConf.load(
    #     "/data1/xiepengyuan/workspace/TTS/TTS/speaker_encoder/preprocess/config/config_unmix_ar16000_ac1.yaml"),
    #     selected_ids=ARG.selected_ids)
