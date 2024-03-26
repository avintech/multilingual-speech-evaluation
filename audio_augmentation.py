from tqdm import tqdm
import pandas as pd
import numpy as np
import torchaudio
import torch
import augment
import random
import os

def augment_audio(original_path):
    random_pitch_shift = lambda: np.random.randint(-100, +100)
    random_room_size = lambda: np.random.randint(0, 51)
    output_path = original_path.replace('recordings/','recordings/augmented/')
    augmented_vector = audio_modification(original_path, random_pitch_shift, random_room_size)
    print(output_path)
    torchaudio.save(output_path, augmented_vector, sample_rate = 44100)
    return output_path

def audio_modification(wave_path, random_pitch_shift, random_room_size):
  x, sr = torchaudio.load(wave_path)
  r = random.randint(1,5)

  if r == 1:
    random_pitch_shift_effect = augment.EffectChain().pitch("-q", random_pitch_shift).rate(sr)
    y = random_pitch_shift_effect.apply(x, src_info={'rate': sr})
  elif r == 2:
    random_reverb = augment.EffectChain().reverb(50, 50, random_room_size).channels(1)
    y = random_reverb.apply(x, src_info={'rate': sr})
  elif r == 3:
    noise_generator = lambda: torch.zeros_like(x).uniform_()
    y = augment.EffectChain().additive_noise(noise_generator, snr=15).apply(x, src_info={'rate': sr})
  else:
    y = augment.EffectChain().time_dropout(max_seconds=0.5).apply(x, src_info={'rate': sr})

  return y