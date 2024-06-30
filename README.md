# WeTTS

## 流程
1. 从剪映或魔音工坊获取数据
2. 转码成单通道 44.1k
3. clip_audio.py，对音频切片
4. ffmpeg.py 转码
5. vits和tools软链接
6. run.sh修改