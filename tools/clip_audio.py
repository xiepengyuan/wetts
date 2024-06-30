# -*- coding: utf-8 -*-

import torch
import webrtcvad
import torchaudio
from vad import frame_generator, vad_pos_collector
import numpy as np
import os
from tqdm import tqdm


def clip_audio(src_root_dir, dst_dir, start_index=1, batch_size=250, num_batch=40, min_sp_ms=1000, max_sp_ms=1500, target_sr=44100):
    names = [f"{str(start_index+i*batch_size).zfill(6)}-{str(start_index + (i+1)*batch_size - 1).zfill(6)}" for i in range(num_batch)]

    for name in names:
        print(name)
        start_index = int(name.split("-")[0])
        audio_path = os.path.join(src_root_dir, f"{name}.wav")

        vad_sr = 48000

        waveform, origin_sr = torchaudio.load(audio_path)
        transform_vad = torchaudio.transforms.Resample(
            orig_freq=origin_sr, new_freq=vad_sr)
        transform_target = torchaudio.transforms.Resample(
            orig_freq=vad_sr, new_freq=target_sr)

        frame_duration_ms = 30
        padding_duration_ms = 300
        vad_samples_per_ms = int(vad_sr / 1000)
        vad_samples_per_frame = vad_samples_per_ms * frame_duration_ms

        waveform = transform_vad(waveform)
        vocals = waveform.detach().cpu().numpy()

        # 转16bit的PCM数据，适配webrtcvad
        vocals_vad = (vocals * (float(1 << 15) - 1)).astype(np.int16).tobytes()
        vad = webrtcvad.Vad(3)
        frames = frame_generator(frame_duration_ms, vocals_vad, vad_sr)
        frames = list(frames)
        poses = vad_pos_collector(vad_sr, frame_duration_ms, padding_duration_ms, vad, frames)
        poses = list(poses)

        tmp_poses = []
        index = int(name.split("-")[0])
        max_index = int(name.split("-")[1])
        for i, pos in tqdm(enumerate(poses)):
            if i == 0:
                tmp_poses.append(pos)
                continue
            # 前一段的最后，和后一段的第一个，相差1s才分开（还要考虑后面是连续的）
            tmp_sp_ms = (pos[0] - tmp_poses[-1][-1]) * frame_duration_ms

            # sp超过最大时长，有问题
            if tmp_sp_ms > max_sp_ms:
                pass
                # print(f"tmp_sp_ms > max_sp_ms, {start_index+i}, {tmp_sp_ms}, {pos[0]} {pos[-1]}")

            # 满足最小时长，输出音频
            if tmp_sp_ms >= min_sp_ms:# or (index==4380 and tmp_sp_ms>=500):
                if i == 1:
                    zero_padding = torch.zeros(1, vad_samples_per_frame*2)
                    audio_seg = waveform[:, tmp_poses[0][0] * vad_samples_per_frame: max(0, (tmp_poses[-1][-1]-7)) * vad_samples_per_frame]
                    audio_seg = torch.cat([zero_padding, audio_seg], dim=1)
                else:
                    audio_seg = waveform[:, max(0, (tmp_poses[0][0]-3)) * vad_samples_per_frame: max(0, (tmp_poses[-1][-1]-7)) * vad_samples_per_frame]
                audio_seg = transform_target(audio_seg)
                filename = f"{str(index).zfill(6)}.wav"
                save_path = os.path.join(dst_dir, filename)
                torchaudio.save(save_path, audio_seg, target_sr)
                # if index == 4380:
                #     for tmp_pos in tmp_poses:
                #         print(tmp_pos[0] * vad_samples_per_frame / 48000, tmp_pos[-1] * vad_samples_per_frame / 48000)
                index += 1
                if index == 13306 or index == 13818:
                    index += 1
                if index > max_index:
                    print(f"error!!!, too many audio slices")
                    break
                tmp_poses = []
            elif tmp_sp_ms < 200:
                pass
            else:
                print(f"maybe wrong, {i+start_index}, {tmp_sp_ms}, {pos[0]} {pos[-1]}")
                pass
            tmp_poses.append(pos)

        # 写入最后一条
        audio_seg = waveform[:, (tmp_poses[0][0]-3) * vad_samples_per_frame: (tmp_poses[-1][-1]-7) * vad_samples_per_frame]
        audio_seg = transform_target(audio_seg)
        filename = f"{str(index).zfill(6)}.wav"
        save_path = os.path.join(dst_dir, filename)
        torchaudio.save(save_path, audio_seg, target_sr)

        if index - start_index + 1 < batch_size:
            print("error! not enough")
            break


def run():
    is_local = False

    if is_local:
        src_root_dir = "E:\\tmp2\\2023-06-09\\moxiaoqi"
        dst_root_dir = "E:\\tmp2\\2023-06-09\\moxiaoqi_splited"
        dst_dir = dst_root_dir
    else:
        src_root_dir = f"/data1/xiepengyuan/audio/tts/tongtong/tongtong_batch_wav"
        dst_root_dir = f"/data1/xiepengyuan/audio/tts/tongtong/tongtong_wav"
        dst_dir = dst_root_dir
    os.makedirs(dst_dir, exist_ok=True)
    clip_audio(src_root_dir, dst_root_dir, start_index=9001, batch_size=1000, num_batch=1, min_sp_ms=1070, max_sp_ms=1500)


def run_test():
    speaker = "moxiaoyun-moyin"
    is_local = False

    if is_local:
        src_root_dir = "E:\\tmp2\\2023-06-09\\moxiaoqi"
        dst_root_dir = "E:\\tmp2\\2023-06-09\\moxiaoqi_splited"
        dst_dir = dst_root_dir
    else:
        src_root_dir = f"/data1/xiepengyuan/audio/tts/tongtong/tongtong_batch_wav"
        dst_root_dir = f"/data1/xiepengyuan/audio/tts/tongtong/tongtong_wav"
        dst_dir = dst_root_dir
    os.makedirs(dst_dir, exist_ok=True)
    clip_audio(src_root_dir, dst_root_dir, batch_size=1000, num_batch=10, min_sp_ms=800, max_sp_ms=1500, target_sr=22050)


if __name__ == '__main__':
    run()
