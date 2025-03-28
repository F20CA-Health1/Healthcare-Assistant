import argparse
import os
import time
import pygame
from gtts import gTTS
from pydub import AudioSegment

# United Nations languages + most spoken languages
DEFAULT_LANGUAGES = ["en", "zh-cn", "es", "fr", "ru", "ar", "ja", "ko", "hi"]

HELP_DESCRIPTION = """
🌍 This program converts text to speech using Google Text-to-Speech (gTTS). 
It supports multiple languages, allows saving and playing audio files, and provides various customization options.

🇨🇳 本程序使用 Google 文字转语音 (gTTS) 将文本转换为语音。
它支持多种语言，可保存和播放音频文件，并提供各种自定义选项。

🇭🇰/🇲🇴 本程式使用 Google 文字轉語音 (gTTS) 將文字轉換為語音。
它支援多種語言，可儲存與播放音訊檔案，並提供各種自訂選項。

🇪🇸 Este programa convierte texto en voz usando Google Text-to-Speech (gTTS).
Soporta múltiples idiomas, permite guardar y reproducir archivos de audio y ofrece diversas opciones de personalización.

🇫🇷 Ce programme convertit du texte en parole à l'aide de Google Text-to-Speech (gTTS).
Il prend en charge plusieurs langues, permet d'enregistrer et de lire des fichiers audio et offre diverses options de personnalisation.

🇷🇺 Эта программа преобразует текст в речь с помощью Google Text-to-Speech (gTTS).
Она поддерживает несколько языков, позволяет сохранять и воспроизводить аудиофайлы и предлагает различные параметры настройки.

🇸🇦 يقوم هذا البرنامج بتحويل النص إلى كلام باستخدام Google Text-to-Speech (gTTS).
وهو يدعم لغات متعددة، ويسمح بحفظ وتشغيل ملفات الصوت، ويقدم خيارات تخصيص متنوعة.

🇯🇵 このプログラムは、Google Text-to-Speech（gTTS）を使用してテキストを音声に変換します。
複数の言語に対応し、音声ファイルの保存や再生が可能で、さまざまなカスタマイズオプションを提供します。

🇰🇷 이 프로그램은 Google 텍스트 음성 변환(gTTS)을 사용하여 텍스트를 음성으로 변환합니다.
여러 언어를 지원하며, 오디오 파일 저장 및 재생이 가능하며, 다양한 맞춤 설정 옵션을 제공합니다。

🇮🇳 यह प्रोग्राम Google Text-to-Speech (gTTS) का उपयोग करके टेक्स्ट को आवाज में बदलता है।
यह कई भाषाओं का समर्थन करता है, ऑडियो फाइलें सहेजने और चलाने की सुविधा देता है और विभिन्न अनुकूलन विकल्प प्रदान करता है।
"""

ARG_HELP_TEXTS = {
    "-t / --text": """
[文本输入] - 要转换为语音的文本。
默认值: 多语言演示文本。

[Text Input] - The text to be converted into speech.
Default: Multi-language demo speech.

[Texte] - Texte à convertir en parole.
Défaut : Discours de démonstration multilingue.

[Texto] - Texto para convertir en voz.
Predeterminado: discurso de demostración multilingüe.

[Текст] - Текст для преобразования в речь.
По умолчанию: демонстрационная речь.

[النص] - النص المراد تحويله إلى كلام.
الافتراضي: خطاب توضيحي.

[テキスト] - 音声に変換するテキスト。
デフォルト: 多言語デモ。

[텍스트] - 음성으로 변환할 텍스트.
기본값: 다국어 데모.
""",
    "-l / --lang": """
[语言选择] - 语音的语言。
默认: 播放所有支持的语言。

[Language] - The language of speech.
Default: Play all supported languages.

[Langue] - Langue de la parole.
Défaut : Lire toutes les langues.

[Idioma] - Idioma de la voz.
Predeterminado: Reproducir todos los idiomas.

[Язык] - Язык речи.
По умолчанию: воспроизвести все языки.

[اللغة] - لغة الكلام.
الافتراضي: تشغيل جميع اللغات.

[言語] - 音声の言語。
デフォルト: すべての言語を再生。

[언어] - 음성의 언어.
기본값: 모든 언어 재생.
""",
    "--speed": """
[语速倍率] - 控制语音播放速度。
默认值: 1.2 | 允许范围: 0.5 - 2.0

[Speech Speed] - Controls the speech playback speed.
Default: 1.2 | Range: 0.5 - 2.0

[Vitesse de la parole] - Contrôle la vitesse de lecture de la parole.
Par défaut: 1.2 | Plage: 0.5 - 2.0

[Velocidad de voz] - Controla la velocidad de reproducción de voz.
Predeterminado: 1.2 | Rango: 0.5 - 2.0

[Скорость речи] - Контролирует скорость воспроизведения речи.
По умолчанию: 1.2 | Диапазон: 0.5 - 2.0

[سرعة الكلام] - يتحكم في سرعة تشغيل الكلام.
الافتراضي: 1.2 | النطاق: 0.5 - 2.0

[話速] - 音声再生速度を制御。
デフォルト: 1.2 | 範囲: 0.5 - 2.0

[음성 속도] - 음성 재생 속도 제어.
기본값: 1.2 | 범위: 0.5 - 2.0
"""
}

def get_help_text():
    """Generate detailed help text with multi-language support."""
    help_text = HELP_DESCRIPTION + "\n\n"
    for arg, translations in ARG_HELP_TEXTS.items():
        help_text += f"\n{arg}\n{translations}\n"
    return help_text

# Expanded language alias dictionary
LANGUAGE_ALIASES = {
    "en": ["en", "english", "ENG", "Eng", "eng", "美语", "英語", "英语"],
    "zh-cn": ["zh-cn", "zh", "cn", "chinese", "Chinese", "CN", "ZH", "ZH-CN", "中文", "汉语", "普通话", "简体", "繁體", "國語", "国语"],
    "es": ["es", "spanish", "español", "ESP", "西班牙语", "西語", "西语"],
    "fr": ["fr", "french", "français", "法语", "法文"],
    "ru": ["ru", "russian", "русский", "俄语", "俄文"],
    "ar": ["ar", "arabic", "عربي", "العربية", "阿拉伯语", "Arab"],
    "ja": ["jp", "ja", "japanese", "日本語", "にほんご", "nihon", "Nihon", "nihongo", "Nihongo", "にっぽんご", "ニホンゴ", "ニッポンゴ", "Nippongo"],
    "ko": ["ko", "korean", "한국어", "조선말", "Korea", "韩语", "朝鲜语", "조선어"],
    "hi": ["hi", "hindi", "हिन्दी", "印度语", "印地語", "हिंदी"],
}

# Default test texts in different languages
DEFAULT_TEXTS = {
    "en": "This is a default test speech in English.",
    "zh-cn": "这是一个默认的测试语音。",
    "es": "Este es un discurso de prueba predeterminado en español.",
    "fr": "Ceci est un discours de test par défaut en français.",
    "ru": "Это стандартная тестовая речь на русском языке.",
    "ar": "هذا خطاب اختبار افتراضي باللغة العربية.",
    "ja": "これはデフォルトのテスト音声です。",
    "ko": "이것은 기본 테스트 음성입니다.",
    "hi": "यह हिंदी में एक डिफ़ॉल्ट परीक्षण भाषण है।",
}

# Supported audio formats
SUPPORTED_FORMATS = ["mp3", "flac", "wav", "ogg"]

def normalize_language(lang_input):
    """Normalize language input."""
    lang_input = lang_input.strip().lower()
    for standard_lang, aliases in LANGUAGE_ALIASES.items():
        if lang_input in [alias.lower() for alias in aliases]:
            return standard_lang
    print(f"[WARNING] Unsupported language input '{lang_input}'. Defaulting to multi-language mode.")
    return "default"

def text_to_speech(text, lang='en', output_dir=".", speed=1.2, play_audio=True, save_audio=True, file_format="mp3"):
    """Convert text to speech using gTTS, save as an audio file, and play it."""
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"tts_{lang}_{timestamp}.{file_format}")

        # Generate speech
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(output_file)

        # Modify playback speed using pydub
        if speed != 1.0:
            audio = AudioSegment.from_file(output_file)
            audio = audio.speedup(playback_speed=speed)
            audio.export(output_file, format=file_format)

        print(f"[INFO] Audio file saved: {output_file}")

        # Play audio
        if play_audio:
            pygame.mixer.quit()
            pygame.mixer.init(frequency=22050)
            pygame.mixer.music.load(output_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

        return output_file if save_audio else None

    except Exception as e:
        print(f"[ERROR] Failed to convert text to speech: {e}")
        return None

def generate_combined_audio(output_dir, loop_count=3, speed=1.4):
    """Generate a combined audio file with all default languages, then play it in a loop."""
    combined_audio = AudioSegment.silent(duration=0)
    all_files = []

    for lang in DEFAULT_LANGUAGES:
        text = DEFAULT_TEXTS.get(lang, "Default test speech.")
        audio_file = text_to_speech(text, lang=lang, output_dir=output_dir, speed=speed, play_audio=False, save_audio=True)
        if audio_file:
            all_files.append(audio_file)

    for file in all_files:
        audio = AudioSegment.from_file(file)
        combined_audio += audio

    combined_output = os.path.join(output_dir, "combined_audio.mp3")
    combined_audio.export(combined_output, format="mp3")
    print(f"[INFO] Combined audio file saved: {combined_output}")

    for file in all_files:
        os.remove(file)
        print(f"[INFO] Deleted: {file}")

    pygame.mixer.quit()
    pygame.mixer.init(frequency=22050)
    for i in range(loop_count):
        print(f"[INFO] Playing combined audio (Loop {i+1}/{loop_count})...")
        pygame.mixer.music.load(combined_output)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

def main():
    """Command-line interface for text-to-speech conversion."""
    parser = argparse.ArgumentParser(
        description=get_help_text(),  # 🚀 直接使用 `get_help_text()` 作为帮助文本
        formatter_class=argparse.RawTextHelpFormatter  # ✅ 确保 `help` 格式正确
    )

    # CLI arguments
    parser.add_argument("-t", "--text", type=str, help="Input text to convert to speech.")
    parser.add_argument("-l", "--lang", type=str, help="Language of the speech (leave empty to use all languages).")
    parser.add_argument("-o", "--output_dir", type=str, default=".", help="Output directory for the audio file.")
    parser.add_argument("-p", "--no-play", action="store_true", help="Disable automatic audio playback.")
    parser.add_argument("-s", "--no-save", action="store_true", help="Disable saving the audio file.")
    parser.add_argument("-f", "--format", type=str, default="mp3", choices=SUPPORTED_FORMATS, help="Audio format (default: mp3).")
    parser.add_argument("--speed", type=float, default=1.2, help="Playback speed multiplier (default: 1.2).")

    args = parser.parse_args()

    # 🚀 如果用户调用 `python tts.py -h`，`argparse` 自动打印帮助并退出，无需手动处理

    # Default multi-language mode
    if args.lang is None:
        print("[INFO] No language specified. Playing all default languages.")
        generate_combined_audio(output_dir=args.output_dir, speed=args.speed)
        return

    # Normalize language input
    lang = normalize_language(args.lang)

    # Set default text if not provided
    text = args.text if args.text else DEFAULT_TEXTS.get(lang, DEFAULT_TEXTS["en"])
    print(f"[INFO] Using language: {lang}, Text: {text}")

    # Convert text to speech
    text_to_speech(
        text=text,
        lang=lang,
        output_dir=args.output_dir,
        speed=args.speed,
        play_audio=not args.no_play,
        save_audio=not args.no_save,
        file_format=args.format
    )

if __name__ == "__main__":
    main()
