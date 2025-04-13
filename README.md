# 环境安装过程
    pip install --upgrade pip
    pip install -U openai-whisper
    # 安装音频处理依赖
    pip install ffmpeg-python pydub
    # 安装离线翻译库 Argos Translate
    pip install argostranslate
    # 安装字幕文件生成库
    pip install srt


# 问题记录
1. 下载whisper失败
    解决方案：改用pip安装, pip install -U openai-whisper
2. whisper下载模型失败
    解决方案：Whisper 的命令行工具并不支持 --download-model 和 --model-dir, 使用 whisper.load_model() 函数时，设置 download_root 参数。
    import whisper
    model = whisper.load_model("medium", download_root="E:/whisper_models")

3. 下载Argos Translate的翻译模型失败
    解决方案：Argos Translate 自带了获取可用模型列表并下载的功能
    ```
    from argostranslate import package
    
    # 1. 获取所有可用的模型包列表
    available_packages = package.get_available_packages()

    # 2. 找到 English→Chinese 的模型
    en_zh_pkg = next(
        p for p in available_packages
        if p.from_code == "en" and p.to_code == "zh"
    )

    print(f"Found package: {en_zh_pkg.name}, version {en_zh_pkg.version}")

    # 3. 下载模型，返回本地文件路径（一个 .zip 文件）
    model_path = en_zh_pkg.download()
    print(f"Downloaded to: {model_path}")

    # 4. 安装模型
    package.install_from_path(model_path)
    print("安装完成！")
    ```

# 生成中文字幕
    1. 单个视频文件生成字幕
    python generate_chs_subtitles.py "1.mp4" --model medium
    2. 文件夹下所有视频文件生成字幕
    python batch_generate_chs_subtitles.py "E:\jigsaw_lesson" --model medium