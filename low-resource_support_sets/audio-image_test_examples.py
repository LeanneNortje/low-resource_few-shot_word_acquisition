#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import torchaudio
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import librosa
import scipy
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

BACKEND = "nccl"
INIT_METHOD = "tcp://localhost:54321"

scipy_windows = {
    'hamming': scipy.signal.hamming,
    'hann': scipy.signal.hann, 
    'blackman': scipy.signal.blackman,
    'bartlett': scipy.signal.bartlett
    }

def preemphasis(signal,coeff=0.97):  
    # function adapted from https://github.com/dharwath
    
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

def LoadAudio(path, alignment, audio_conf):
    threshold = 0
    audio_type = audio_conf.get('audio_type')
    if audio_type not in ['melspectrogram', 'spectrogram']:
        raise ValueError('Invalid audio_type specified in audio_conf. Must be one of [melspectrogram, spectrogram]')

    preemph_coef = audio_conf.get('preemph_coef')
    sample_rate = audio_conf.get('sample_rate')
    window_size = audio_conf.get('window_size')
    window_stride = audio_conf.get('window_stride')
    window_type = audio_conf.get('window_type')
    num_mel_bins = audio_conf.get('num_mel_bins')
    target_length = audio_conf.get('target_length')
    fmin = audio_conf.get('fmin')
    n_fft = audio_conf.get('n_fft', int(sample_rate * window_size))
    win_length = int(sample_rate * window_size)
    hop_length = int(sample_rate * window_stride)

    # load audio, subtract DC, preemphasis
    y, sr = librosa.load(path, sample_rate)
    wav = y
    dur = librosa.get_duration(y=y, sr=sr)
    nsamples = y.shape[0]
    if y.size == 0:
        y = np.zeros(target_length)
    y = y - y.mean()
    y = preemphasis(y, preemph_coef)

    # compute mel spectrogram / filterbanks
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=scipy_windows.get(window_type, scipy_windows['hamming']))
    spec = np.abs(stft)**2 # Power spectrum
    if audio_type == 'melspectrogram':
        mel_basis = librosa.filters.mel(sr, n_fft, n_mels=num_mel_bins, fmin=fmin)
        melspec = np.dot(mel_basis, spec)
        logspec = librosa.power_to_db(melspec, ref=np.max)
    elif audio_type == 'spectrogram':
        logspec = librosa.power_to_db(spec, ref=np.max)
    # n_frames = logspec.shape[1]
    logspec = torch.FloatTensor(logspec[:, np.maximum(alignment[0]- threshold, 0): np.minimum(alignment[1] + threshold, nsamples)])
    nsamples = logspec.size(1)

    return wav, torch.tensor(logspec), nsamples#, n_frames

def myRandomCrop(im, resize, to_tensor):

        im = resize(im)
        im = to_tensor(im)
        return im

def LoadImage(impath, resize, image_normalize, to_tensor):
    img = Image.open(impath).convert('RGB')
    # img = self.image_resize_and_crop(img)
    img = myRandomCrop(img, resize, to_tensor)
    # img = image_normalize(img)
    return img

image_base = Path('../Datasets/spokencoco')
episodes = np.load(Path("data/test_episodes.npz"), allow_pickle=True)['episodes'].item()

vocab = []
with open('data/test_keywords.txt', 'r') as f:
    for keyword in f:
        vocab.append(' '.join(keyword.split()))

alignments = {}
prev = ''
prev_wav = ''
prev_start = 0
with open(Path('../Datasets/spokencoco/SpokenCOCO/words.txt'), 'r') as f:
    for line in f:
        wav, start, stop, label = line.strip().split()
        if label in vocab or (label == 'hydrant' and prev == 'fire' and wav == prev_wav):
            if wav not in alignments: alignments[wav] = {}
            if label == 'hydrant' and prev == 'fire': 
                label = prev + " " + label
                start = prev_start
            if label not in alignments[wav]: alignments[wav][label] = (int(float(start)*100), int(float(stop)*100))
        prev = label
        prev_wav = wav
        prev_start = start

image_conf = {
    "crop_size": 224,
    "center_crop": False,
    "RGB_mean": [0.485, 0.456, 0.406],
    "RGB_std": [0.229, 0.224, 0.225]
}

audio_conf = {
    "audio_type": "melspectrogram",
    "preemph_coef": 0.97,
    "sample_rate": 16000,
    "window_size": 0.025,
    "window_stride": 0.01,
    "window_type": "hamming",
    "num_mel_bins": 40,
    "target_length": 1024,
    "use_raw_length": False,
    "padval": 0,
    "fmin": 20
}

target_length = audio_conf.get('target_length', 1024)
padval = audio_conf.get('padval', 0)
crop_size = image_conf.get('crop_size')
center_crop = image_conf.get('center_crop')
RGB_mean = image_conf.get('RGB_mean')
RGB_std = image_conf.get('RGB_std')
resize = transforms.Resize((256, 256))
to_tensor = transforms.ToTensor()
image_normalize = transforms.Normalize(mean=RGB_mean, std=RGB_std)

image_resize = transforms.transforms.Resize((256, 256))
trans = transforms.ToPILImage()

save_fn = Path('audio-image_test_examples')
save_fn.mkdir(parents=True, exist_ok=True)

episode_names = list(episodes.keys())
episode_names.remove('matching_set')
m = 3
for episode_num in tqdm(episode_names[0:100]):

    episode = episodes[episode_num]
    for word in episode['queries']:
        wav, spkr = episode['queries'][word]
        fn = image_base / 'SpokenCOCO' / wav
        lookup = str(Path(wav).stem)
        wave, og, frames = LoadAudio(fn, alignments[lookup][word], audio_conf)
        downsampled = np.zeros((wave.shape[0]//m, 1))
        c = 0
        for i in range(downsampled.shape[0]):
            for j in range(m):
                downsampled[i, :] += wave[c+j]
            c += m
        plt.figure()
        plt.plot(downsampled, c='olivedrab')
        plt.axis('off')
        plt.savefig(save_fn / Path(f'{word}_audio_{episode_num}.jpg'), bbox_inches='tight',pad_inches = 0)

        aud = torch.zeros((og.size(0)//m, og.size(1)), device=og.device)
        c = 0
        for i in range(aud.size(0)):
            for j in range(m):
                aud[i, :] += og[c+j, :]
            c += m
        plt.figure()
        plt.imshow(aud.cpu().detach().numpy(), interpolation = 'nearest')
        plt.axis('off')
        plt.savefig(save_fn / Path(f'{word}_spectogram_{episode_num}.jpg'), bbox_inches='tight',pad_inches = 0)

        fn = image_base / episode['matching_set'][word]
        im = LoadImage(fn, resize, image_normalize, to_tensor)
        plt.figure()
        plt.imshow(trans(im))
        plt.axis('off')
        plt.savefig(save_fn / Path(f'{word}_image_{episode_num}.jpg'), bbox_inches='tight',pad_inches = 0)