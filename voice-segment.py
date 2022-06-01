import librosa
import numpy as np

from pydub import AudioSegment

from librosa import feature
from librosa import core

VOICE_FILE = '/mnt/c/Users/88698/Desktop/單字聽力/單字聽力/1 核心單字基本學習/核心單字基本學習 Day21.mp3'

def to_normalized_array(audio_chunk, fs, librosa_fs):
   samples = audio_chunk.get_array_of_samples()
   arr = np.array(samples).astype(np.float32) / np.iinfo(np.int16).max
   return librosa.core.resample(arr, orig_sr=fs, target_sr=librosa_fs)


audio_chunk = AudioSegment.from_mp3(VOICE_FILE)
audio_chunk = audio_chunk.set_sample_width(2).set_channels(1).set_frame_rate(44100)

fs = 44100
librosa_fs = 16000
top_db = 40

arr = to_normalized_array(audio_chunk, fs, librosa_fs)

mse = feature.rms(y=arr, frame_length=2048, hop_length=512) ** 2
mse_db = core.power_to_db(mse.squeeze(), top_db=None)

percentile_parameter = 0.1     # [%]
extra_db_parameter = 5         # [dB]
threshold = np.percentile(mse_db, percentile_parameter ) + extra_db_parameter

edges = librosa.effects.split(arr, top_db=threshold) / librosa_fs

print(len(edges))