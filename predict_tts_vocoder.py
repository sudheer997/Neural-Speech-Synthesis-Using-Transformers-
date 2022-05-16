from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from Transformer_tts.model.factory import tts_ljspeech
from Transformer_tts.data.audio import Audio
from hifigan_predictor import HiFiGANPredictor

import sys
sys.path.append("Transformer_tts")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--text', '-t', dest='text', default=None, type=str)
    parser.add_argument('--outdir', '-o', dest='outdir', default=None, type=str)
    parser.add_argument('--store_mel', '-m', dest='store_mel', action='store_true')
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true')
    parser.add_argument('--voc', dest='vocoder', default='glim')
    args = parser.parse_args()

    if args.text is not None:
        text = [args.text]
        fname = 'custom_text'
    else:
        fname = None
        text = None
        print(f'Specify either an input text (-t "some text") or a text input file (-f /path/to/file.txt)')
        exit()

    # load the appropriate model
    outdir = Path(args.outdir) if args.outdir is not None else Path('.')

    model = tts_ljspeech()
    output_file_name = f'{fname}_{args.vocoder}_ljspeech_v1'

    if args.vocoder == 'hifigan':
        vocoder = HiFiGANPredictor.from_folder('hifigan/en')
    outdir = outdir / 'outputs'
    outdir.mkdir(exist_ok=True, parents=True)
    audio = Audio.from_config(model.config)
    print(f'Output wav under {outdir}')
    wavs = []
    print(text)
    for i, text_line in enumerate(text):
        print(text_line)
        phons = model.text_pipeline.phonemizer(text_line)
        tokens = model.text_pipeline.tokenizer(phons)
        if args.verbose:
            print(f'Predicting {text_line}')
            print(f'Phonemes: "{phons}"')
            print(f'Tokens: "{tokens}"')
        out = model.predict(tokens, encode=False, phoneme_max_duration=None)
        mel = out['mel'].numpy().T
        if args.vocoder == 'glim':
            wav = audio.reconstruct_waveform(mel)
        else:
            wav = vocoder([mel])[0]
        wavs.append(wav)
        if args.store_mel:
            np.save((outdir / (output_file_name + f'_{i}')).with_suffix('.mel'), out['mel'].numpy())
    audio.save_wav(np.concatenate(wavs), (outdir / output_file_name).with_suffix('.wav'))