# -*- coding: utf-8 -*-

import os
import sys
import argparse
import tempfile
import math
import time
from pathlib import Path

import ffmpeg
import srt
from faster_whisper import WhisperModel
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# --- 1. 全局模型加载 ---
# 将模型加载放在全局，避免重复加载
print("Initializing translation model (facebook/m2m100_418M)...")
TRANSLATION_TOKENIZER = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
TRANSLATION_MODEL = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
print("Translation model loaded.")

# --- 2. 术语修正字典 ---
# 按语言对组织，方便扩展
TERMINOLOGY_CORRECTIONS = {
    "en": {
        "order flu": "order flow",
        "liquidity pull": "liquidity pool",
        "us the": "UST",
        "btc": "BTC",
    },
    "zh": {
        # 通用的中文修正
        "订购流": "订单流",
        "市场制造者": "做市商",
        "点差": "价差",
        # 特定于英文源的中文修正
        "流动性池": "流动性资金池", # 示例
    },
    "ja": {
        # 如果需要，可以添加日语术语修正
        # "ある特定の日本語": "修正後の日本語",
    }
}

# --- 3. 核心功能函数 ---

def translate_text(text, src_lang="en", tgt_lang="zh"):
    """
    使用 M2M100 模型进行通用翻译。
    """
    TRANSLATION_TOKENIZER.src_lang = src_lang
    encoded = TRANSLATION_TOKENIZER(text, return_tensors="pt")
    generated_tokens = TRANSLATION_MODEL.generate(
        **encoded,
        forced_bos_token_id=TRANSLATION_TOKENIZER.get_lang_id(tgt_lang)
    )
    translated = TRANSLATION_TOKENIZER.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated

def apply_corrections(text, lang_code):
    """
    根据语言代码应用相应的术语修正。
    """
    if lang_code in TERMINOLOGY_CORRECTIONS:
        for wrong, correct in TERMINOLOGY_CORRECTIONS[lang_code].items():
            text = text.replace(wrong, correct)
    return text

def transcribe_audio_chunks(audio_path, whisper_model, lang_code, chunk_length_s=30):
    """
    将音频文件分块转写为指定语言的文本。
    """
    try:
        info = ffmpeg.probe(audio_path)
        duration_s = float(info["format"]["duration"])
    except ffmpeg.Error as e:
        print(f"Error probing audio file: {e.stderr}", file=sys.stderr)
        return []

    all_segments = []
    n_chunks = math.ceil(duration_s / chunk_length_s)
    print(f"  Transcribing in {n_chunks} chunks...")

    for i in range(n_chunks):
        t1 = time.time()
        start_time = i * chunk_length_s
        
        # 使用 with atexit 注册临时文件删除，更安全
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            (
                ffmpeg
                .input(audio_path, ss=start_time, t=chunk_length_s)
                .output(tmp_path, ac=1, ar=16000, format="wav") # 使用标准 SAMPLE_RATE
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )

            # 为日语转写指定语言代码
            segments, _ = whisper_model.transcribe(tmp_path, language=lang_code)
            
            for seg in segments:
                all_segments.append({
                    "start": seg.start + start_time,
                    "end": seg.end + start_time,
                    "text": seg.text
                })

        except ffmpeg.Error as e:
            print(f"  Error processing chunk {i}: {e.stderr.decode()}", file=sys.stderr)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        t2 = time.time()
        print(f"  - Chunk {i+1}/{n_chunks} processed in {t2-t1:.2f} seconds.")

    all_segments.sort(key=lambda x: x["start"])
    return all_segments

def process_video(video_path: Path, whisper_model, src_lang, chunk_length_s):
    """
    处理单个视频文件，生成目标语言字幕。
    """
    print("-" * 50)
    print(f"→ Processing: {video_path.name}")

    # 1. 路径设置
    srt_path_en = video_path.with_suffix(".en.srt")
    srt_path_bilingual = video_path.with_suffix(f".{src_lang}-zh.srt")
    
    if srt_path_bilingual.exists():
        print(f"→ Skipping: Bilingual subtitles already exist for {video_path.name}\n")
        return

    # 2. 音频提取
    audio_path = video_path.with_suffix("._temp_audio.wav")
    print("  Extracting audio...")
    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(str(audio_path), ac=1, ar=16000, format="wav") # 16000Hz is Whisper's sample rate
            .overwrite_output()
            .run(quiet=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print(f"  Audio extraction failed: {e.stderr.decode()}", file=sys.stderr)
        return

    # 3. 音频转写 (源语言)
    print(f"  Transcribing audio to {src_lang.upper()} text...")
    segments = transcribe_audio_chunks(str(audio_path), whisper_model, src_lang, chunk_length_s)

    if not segments:
        print("  Transcription failed or produced no segments.", file=sys.stderr)
        if audio_path.exists():
            os.remove(audio_path)
        return

    # 4. 翻译与字幕生成
    print(f"  Translating from {src_lang.upper()} to Chinese and generating subtitles...")
    subs = []
    for i, seg in enumerate(segments, start=1):
        # 4.1 获取并修正源语言文本
        src_text = seg["text"].strip()
        src_text = apply_corrections(src_text, src_lang)

        # 4.2 翻译为中文
        zh_text = translate_text(src_text, src_lang=src_lang, tgt_lang="zh")
        
        # 4.3 修正中文翻译
        zh_text = apply_corrections(zh_text, "zh")
        
        # 4.4 创建字幕内容
        if src_lang == "en":
            # 对于英文，保留中英双语
            bilingual_text = f"{src_text}\n{zh_text}"
        else:
            # 对于其他语言（如日语），只保留中文字幕
            bilingual_text = zh_text

        subs.append(srt.Subtitle(
            index=i,
            start=srt.timedelta(seconds=seg["start"]),
            end=srt.timedelta(seconds=seg["end"]),
            content=bilingual_text
        ))

    # 5. 写入 SRT 文件
    print(f"  Writing subtitles to {srt_path_bilingual.name}...")
    with open(srt_path_bilingual, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

    # 6. 清理临时文件
    if audio_path.exists():
        os.remove(audio_path)
    
    t_end = time.time()
    print(f"  ✔ Finished. Subtitles generated: {srt_path_bilingual.name}")
    print("-" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch generate subtitles for video files. Supports EN->ZH-EN and JA->ZH.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Root directory to search for video files (.mp4, .mkv, etc.)."
    )
    parser.add_argument(
        "--lang", "-l",
        type=str,
        default="en",
        choices=["en", "ja"],
        help=(
            "Source language of the videos.\n"
            "  'en': English video -> English + Chinese subtitles.\n"
            "  'ja': Japanese video -> Chinese subtitles."
        )
    )
    parser.add_argument(
        "--model", "-m",
        default="small",
        help="Whisper model name (e.g., tiny, base, small, medium, large-v3)."
    )
    parser.add_argument(
        "--chunk-length", "-c",
        type=int,
        default=180,
        help="Length of audio chunks in seconds for transcription."
    )
    parser.add_argument(
        "--file-ext", "-e",
        type=str,
        default="mp4",
        help="Video file extension to process (e.g., mp4, mkv)."
    )
    args = parser.parse_args()

    # 加载 Whisper 模型
    # 对于多语言，不应使用 .en 后缀的模型
    model_name = args.model
    if args.lang != 'en' and model_name.endswith('.en'):
        model_name = model_name[:-3]
        print(f"Warning: Switched to multilingual model '{model_name}' for '{args.lang}' language processing.")
    
    print(f"Loading Whisper model: {model_name}...")
    try:
        whisper_model = WhisperModel(model_name, device="cuda", compute_type="float16")
    except Exception as e:
        print(f"Error loading Whisper model: {e}", file=sys.stderr)
        print("Please ensure CUDA is available and dependencies are installed correctly.", file=sys.stderr)
        sys.exit(1)
    print("Whisper model loaded.")

    # 遍历目录，批量处理
    input_path = Path(args.input_dir)
    video_files = list(input_path.rglob(f"*.{args.file_ext.lower()}"))
    video_files.extend(list(input_path.rglob(f"*.{args.file_ext.upper()}")))

    if not video_files:
        print(f"No '.{args.file_ext}' files found in '{args.input_dir}'.")
        return

    print(f"\nFound {len(video_files)} video file(s) to process.\n")
    
    for video_path in video_files:
        process_video(
            video_path,
            whisper_model,
            src_lang=args.lang,
            chunk_length_s=args.chunk_length
        )

    print("All tasks completed!")


if __name__ == "__main__":
    main()