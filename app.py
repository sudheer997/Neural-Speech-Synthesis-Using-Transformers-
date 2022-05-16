import numpy as np
import streamlit as st
import os
import time
import glob
import os

from Transformer_tts.data.audio import Audio
from Transformer_tts.model.factory import tts_ljspeech

#
# from gtts import gTTS
# from googletrans import Translator
from hifigan_predictor import HiFiGANPredictor


st.title("Text to speech")

text = st.text_input("Enter text")
vocoder = st.selectbox(
    "Select the vocoder",
    ("HiFi-GAN", "Griffin-Lim"),
)

speed = st.selectbox(
    "Speed",
    (0.25, 0.75, 1, 1.25, 1.5, 1.75, 2.0)
)


def text_to_speech(text, vocoder, speed):
    print(text, vocoder, speed, type(speed))
    hifigan_vocoder = HiFiGANPredictor.from_folder('hifigan/en')
    model = tts_ljspeech()
    audio = Audio.from_config(model.config)
    # global hifigan_vocoder
    output = []
    for i, text_line in enumerate([text]):
        print(text_line)
        phons = model.text_pipeline.phonemizer(text_line)
        tokens = model.text_pipeline.tokenizer(phons)
        # print(f'Predicting {text_line}')
        # print(f'Phonemes: "{phons}"')
        # print(f'Tokens: "{tokens}"')
        out = model.predict(tokens, encode=False, phoneme_max_duration=None, speed_regulator=speed)
        mel = out['mel'].numpy().T
        if vocoder == 'Griffin-Lim':
            wav = audio.reconstruct_waveform(mel)
        else:
            wav = hifigan_vocoder([mel])[0]
        output.append(wav)
    audio.save_wav(np.concatenate(output), "outputs/output.wav")

    return "output"


if st.button("convert"):
    result = text_to_speech(text, vocoder, speed)
    audio_file = open(f"outputs/output.wav", "rb")
    audio_bytes = audio_file.read()
    st.markdown(f"## Your audio:")
    st.audio(audio_bytes, format="audio/mp3", start_time=0)


def remove_files(n):
    mp3_files = glob.glob("temp/*mp3")
    if len(mp3_files) != 0:
        now = time.time()
        n_days = n * 86400
        for f in mp3_files:
            if os.stat(f).st_mtime < now - n_days:
                os.remove(f)
                print("Deleted ", f)


remove_files(7)
