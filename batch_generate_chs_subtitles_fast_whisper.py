# -*- coding: utf-8 -*-

import os
import sys
import argparse
import tempfile
import math
import time
import threading
from queue import Queue

import whisper
import ffmpeg
import srt
from whisper.audio import SAMPLE_RATE
from argostranslate.translate import get_translation_from_codes
from faster_whisper import WhisperModel
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
translate_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")

def m2m100_translate(text, src_lang="en", tgt_lang="zh"):
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = translate_model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))
    translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated

# 用于修正英文术语的字典
TERMINOLOGY_CORRECTIONS_EN = {
    "order flu": "order flow",
    "liquidity pull": "liquidity pool",
    "us the": "UST",
    "btc": "BTC",
    # 更多英文术语修正
}

# 用于修正中文术语的字典
TERMINOLOGY_CORRECTIONS_CN = {
    "订购流": "订单流",
    "市场制造者": "做市商",
    "点差": "价差",
    # 更多中文术语修正
}

def apply_corrections(text, terminology_dict):
    """
    应用术语修正（可以传入不同的修正字典）。
    """
    for wrong, correct in terminology_dict.items():
        text = text.replace(wrong, correct)
    return text

def offline_translate(model, text):
    prompt = f"translate English to Chinese: {text}"
    output = model(prompt)
    return apply_corrections(output.strip(), TERMINOLOGY_CORRECTIONS_CN)

def transcribe_in_chunks(audio_path, model, chunk_length_s=30):
    info = ffmpeg.probe(audio_path)
    duration_s = float(info["format"]["duration"])
    all_segments = []
    n_chunks = math.ceil(duration_s / chunk_length_s)

    for i in range(n_chunks):
        t1 = time.time()
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
        segments, _ = model.transcribe(tmp_path, language="en")
        for seg in segments:
            all_segments.append({
                "start": seg.start + start_time,
                "end": seg.end + start_time,
                "text": seg.text
            })
        os.remove(tmp_path)
        t2 = time.time()
        print(f"已完成分块{i}的推理, 耗时: {int(t2-t1)}秒")
    all_segments.sort(key=lambda x: x["start"])
    return all_segments

def process_video(video_path, model, chunk_length_s=30):
    """
    对单个视频文件生成同目录下的 .srt 字幕。
    若同名 .srt 已存在，则跳过处理。
    """
    t1 = time.time()
    base, _    = os.path.splitext(video_path)
    srt_path   = base + ".srt"

    # 如果字幕已存在，跳过
    if os.path.exists(srt_path):
        print(f"→ 跳过：{video_path} （已存在 {os.path.basename(srt_path)}）\n")
        return

    audio_path = base + "_audio.wav"
    print(f"→ 处理：{video_path}")

    # 1. 提取音频
    print("  提取音频 …")
    (
        ffmpeg
        .input(video_path)
        .output(audio_path, ac=1, ar=SAMPLE_RATE, format="wav")
        .overwrite_output()
        .run(quiet=True)
    )

    # 2. 分块转写
    print("  分块转写(音频转英文) …")
    segments = transcribe_in_chunks(audio_path, model, chunk_length_s)

    # 3. 翻译并组合中英文
    print("  翻译中(英文转中文) …")
    # translation = get_translation_from_codes("en", "zh")
    translations = []
    for seg in segments:
        eng_text = seg["text"].strip()
        # zh_text = translation.translate(eng_text)
        # zh_text = offline_translate(translate_model, eng_text)
        eng_text = apply_corrections(eng_text, TERMINOLOGY_CORRECTIONS_EN)
        zh_text = m2m100_translate(eng_text)
        bilingual_text = f"{eng_text}\n{zh_text}"
        translations.append({
            "start": seg["start"],
            "end":   seg["end"],
            "text":  bilingual_text
        })

    # 4. 写入 SRT
    print("  写入字幕文件 …")
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
    print("  清理临时音频 …")
    try:
        os.remove(audio_path)
    except OSError:
        pass

    t2 = time.time()
    print(f"  已生成字幕：{srt_path}, 耗时: {int(t2-t1)}秒\n")

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
    parser.add_argument(
        "--chunk-length", "-c",
        type=int,
        default=180,
        help="每次处理的音频块长度（秒）"
    )
    args = parser.parse_args()

    # 切换到脚本所在目录，确保相对路径一致
    os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

    # 加载 Whisper 模型（只加载一次）
    print(f"加载 Whisper 模型：{args.model} …")
    # model = whisper.load_model(args.model)
    model = WhisperModel(args.model, device="cuda", compute_type="float16")

    # 遍历目录，批量处理
    for root, _, files in os.walk(args.input_dir):
        for fn in files:
            if fn.lower().endswith(".mp4"):
                video_path = os.path.join(root, fn)
                process_video(video_path, model, chunk_length_s=args.chunk_length)

    print("全部完成！")

if __name__ == "__main__":
    main()
