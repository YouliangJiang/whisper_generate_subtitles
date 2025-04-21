# -*- coding: utf-8 -*-

import os
import sys
import whisper
import ffmpeg
import srt
import argparse
from pydub import AudioSegment
from argostranslate import translate, package
from argostranslate.translate import get_translation_from_codes

# 设置工作目录为脚本目录
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(script_dir)

def extract_audio(video_path, audio_path):
    # 使用 ffmpeg 提取音频为 wav
    (
        ffmpeg
        .input(video_path)
        .output(audio_path, ac=1, ar='16k', format='wav')
        .overwrite_output()
        .run(quiet=True)
    )

def transcribe(audio_path, model):
    result = model.transcribe(audio_path, language="en")
    return result["segments"]

def translate_segments(segments):
    # 直接拿到 English→Chinese 的翻译器
    translation = get_translation_from_codes("en", "zh")
    translations = []
    for seg in segments:
        src = seg["text"].strip()
        # 调用 ITranslation.translate()
        dst = translation.translate(src)
        translations.append({
            "start": seg["start"],
            "end":   seg["end"],
            "text":  dst
        })
    return translations

def segments_to_srt(translations, srt_path):
    subs = []
    for i, seg in enumerate(translations, start=1):
        subs.append(srt.Subtitle(index=i,
                                 start=srt.timedelta(seconds=seg["start"]),
                                 end=srt.timedelta(seconds=seg["end"]),
                                 content=seg["text"]))
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

def mux_subtitle(video_path, srt_path, output_path):
    # 将 .srt 字幕内嵌到 MP4
    (
        ffmpeg
        .input(video_path)
        .output(output_path, vf=f"subtitles={srt_path}:force_style='FontName=Arial,FontSize=24'")
        .overwrite_output()
        .run(quiet=True)
    )

def process_video(video_path, model):
    base, _ = os.path.splitext(video_path)
    audio_path = base + "_audio.wav"
    srt_path   = base + ".srt"

    print(f"→ 处理：{video_path}")
    extract_audio(video_path, audio_path)
    # 现在直接传 model，不会重复 load
    segments = transcribe(audio_path, model)
    translations = translate_segments(segments)
    segments_to_srt(translations, srt_path)
    os.remove(audio_path)
    print(f"  已生成字幕：{srt_path}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="输入 MP4 视频文件")
    parser.add_argument("--model", default="medium", help="Whisper 模型大小")
    args = parser.parse_args()

    video_path = args.video
    base, _    = os.path.splitext(video_path)
    audio_path = base + "_audio.wav"
    srt_path   = base + "_chs.srt"
    output_mp4 = base + "_chs.mp4"

    print("1. 提取音频 …")
    extract_audio(video_path, audio_path)

    print("2. 英文转写 …")
    model = whisper.load_model(args.model)
    segments = transcribe(audio_path, model=model)

    print("3. 翻译成中文 …")
    translations = translate_segments(segments)

    print("4. 生成 SRT 字幕 …")
    segments_to_srt(translations, srt_path)

    # print("5. 合成带字幕视频 …")
    # mux_subtitle(video_path, srt_path, output_mp4)

    print(f"完成！输出文件：{output_mp4}")

if __name__ == "__main__":
    main()
