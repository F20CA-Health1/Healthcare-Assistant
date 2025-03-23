import argparse
import os
import time
import asyncio
import pygame
from edge_tts import Communicate
from pydub import AudioSegment

# United Nations languages + most spoken languages
DEFAULT_LANGUAGES = ["en", "zh-cn", "es", "fr", "ru", "ar", "ja", "ko", "hi"]

HELP_DESCRIPTION = """
🌍 This program converts text to speech using Microsoft Edge-TTS.
It supports multiple languages, allows saving and playing audio files, and provides various customization options.
"""

# Supported audio formats
SUPPORTED_FORMATS = ["mp3", "flac", "wav", "ogg"]

# Edge-TTS language mapping (some languages require specific voices)
EDGE_TTS_VOICES = {
    "en": "en-US-AriaNeural",
    "zh-cn": "zh-CN-XiaoxiaoNeural",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "ar": "ar-SA-ZariyahNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "hi": "hi-IN-MadhurNeural",
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

async def text_to_speech(text, lang="en", output_dir=".", speed=1.2, play_audio=True, save_audio=True, file_format="mp3"):
    """Convert text to speech using Microsoft Edge-TTS, save as an audio file, and play it."""
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"tts_{lang}_{timestamp}.{file_format}")

        # Select appropriate voice
        voice = EDGE_TTS_VOICES.get(lang, "en-US-AriaNeural")
        print(f"[INFO] Using Edge-TTS voice: {voice}")

        # Generate speech
        communicate = Communicate(text, voice)
        await communicate.save(output_file)

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

async def generate_combined_audio(output_dir, loop_count=3, speed=1.4):
    """Generate a combined audio file with all default languages, then play it in a loop."""
    combined_audio = AudioSegment.silent(duration=0)
    all_files = []

    for lang in DEFAULT_LANGUAGES:
        text = DEFAULT_TEXTS.get(lang, "Default test speech.")
        audio_file = await text_to_speech(text, lang=lang, output_dir=output_dir, speed=speed, play_audio=False, save_audio=True)
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
    parser = argparse.ArgumentParser(description=HELP_DESCRIPTION)

    parser.add_argument("-t", "--text", type=str, help="Input text to convert to speech.")
    parser.add_argument("-l", "--lang", type=str, help="Language of the speech (leave empty to use all languages).")
    parser.add_argument("-o", "--output_dir", type=str, default=".", help="Output directory for the audio file.")
    parser.add_argument("-p", "--no-play", action="store_true", help="Disable automatic audio playback.")
    parser.add_argument("-s", "--no-save", action="store_true", help="Disable saving the audio file.")
    parser.add_argument("-f", "--format", type=str, default="mp3", choices=SUPPORTED_FORMATS, help="Audio format (default: mp3).")
    parser.add_argument("--speed", type=float, default=1.2, help="Playback speed multiplier (default: 1.2).")

    args = parser.parse_args()

    if args.lang is None:
        print("[INFO] No language specified. Playing all default languages.")
        asyncio.run(generate_combined_audio(output_dir=args.output_dir, speed=args.speed))
        return

    lang = args.lang.lower()
    if lang not in EDGE_TTS_VOICES:
        print(f"[WARNING] Unsupported language '{lang}'. Defaulting to English.")
        lang = "en"

    text = args.text if args.text else DEFAULT_TEXTS.get(lang, DEFAULT_TEXTS["en"])
    print(f"[INFO] Using language: {lang}, Text: {text}")

    asyncio.run(text_to_speech(
        text=text,
        lang=lang,
        output_dir=args.output_dir,
        speed=args.speed,
        play_audio=not args.no_play,
        save_audio=not args.no_save,
        file_format=args.format
    ))

if __name__ == "__main__":
    main()
