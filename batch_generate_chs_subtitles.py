# -*- coding: utf-8 -*-

import os
import sys
import argparse
import tempfile
import math

import whisper
import ffmpeg
import srt
from whisper.audio import SAMPLE_RATE
from argostranslate.translate import get_translation_from_codes

def transcribe_in_chunks(audio_path, model, chunk_length_s=30):
    """
    将音频按 chunk_length_s 秒分块，用 Whisper 分别转写，
    并把各块的时间戳平移后合并返回完整 segments 列表。
    """
    info = ffmpeg.probe(audio_path)
    duration_s = float(info["format"]["duration"])
    all_segments = []
    n_chunks = math.ceil(duration_s / chunk_length_s)
    for i in range(n_chunks):
        start_time = i * chunk_length_s
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        (
            ffmpeg
            .input(audio_path, ss=start_time, t=chunk_length_s)
            .output(tmp_path, ac=1, ar=SAMPLE_RATE, format="wav")
            .overwrite_output()
            .run(quiet=True)
        )
        result = model.transcribe(tmp_path, language="en")
        for seg in result["segments"]:
            seg["start"] += start_time
            seg["end"]   += start_time
            all_segments.append(seg)
        os.remove(tmp_path)
    all_segments.sort(key=lambda x: x["start"])
    return all_segments

def process_video(video_path, model):
    """
    对单个视频文件生成同目录下的 .srt 字幕。
    若同名 .srt 已存在，则跳过处理。
    """
    base, _    = os.path.splitext(video_path)
    srt_path   = base + ".srt"

    # 如果字幕已存在，跳过
    if os.path.exists(srt_path):
        print(f"→ 跳过：{video_path} （已存在 {os.path.basename(srt_path)}）\n")
        return

    audio_path = base + "_audio.wav"
    print(f"→ 处理：{video_path}")

    # 1. 提取音频
    (
        ffmpeg
        .input(video_path)
        .output(audio_path, ac=1, ar=SAMPLE_RATE, format="wav")
        .overwrite_output()
        .run(quiet=True)
    )

    # 2. 分块转写
    segments = transcribe_in_chunks(audio_path, model, chunk_length_s=30)

    # 3. 翻译
    translation = get_translation_from_codes("en", "zh")
    translations = []
    for seg in segments:
        dst = translation.translate(seg["text"].strip())
        translations.append({
            "start": seg["start"],
            "end":   seg["end"],
            "text":  dst
        })

    # 4. 写入 SRT
    subs = []
    for i, seg in enumerate(translations, start=1):
        subs.append(srt.Subtitle(
            index=i,
            start=srt.timedelta(seconds=seg["start"]),
            end=srt.timedelta(seconds=seg["end"]),
            content=seg["text"]
        ))
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

    # 5. 清理临时音频
    try:
        os.remove(audio_path)
    except OSError:
        pass

    print(f"  已生成字幕：{srt_path}\n")

def main():
    parser = argparse.ArgumentParser(
        description="批量为目录下所有 MP4 视频生成中文字幕（.srt），已存在则跳过"
    )
    parser.add_argument(
        "input_dir",
        help="要处理的根目录（会递归查找 .mp4/.MP4 文件）"
    )
    parser.add_argument(
        "--model", "-m",
        default="small.en",
        help="Whisper 模型名称（如 tiny, base, small, small.en, medium）"
    )
    args = parser.parse_args()

    # 切换到脚本所在目录，确保相对路径一致
    os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

    # 加载 Whisper 模型（只加载一次）
    print(f"加载 Whisper 模型：{args.model} …")
    model = whisper.load_model(args.model)

    # 遍历目录，批量处理
    for root, _, files in os.walk(args.input_dir):
        for fn in files:
            if fn.lower().endswith(".mp4"):
                video_path = os.path.join(root, fn)
                process_video(video_path, model)

    print("全部完成！")

if __name__ == "__main__":
    main()
