#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
# adapted from https://github.com/dharwath

import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel as DDP
from dataloaders import *
from models.setup import *
from models.util import *
from models.GeneralModels import *
from models.multimodalModels import *
from training.util import *
from evaluation.calculations import *
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from training import validate
import time
from tqdm import tqdm

import numpy as trainable_parameters
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import scipy
import scipy.signal
from scipy.spatial import distance
import librosa
import matplotlib.lines as lines

import itertools
import seaborn as sns
from torchvision.io import read_image
from torchvision.models import *
import textgrids

BACKEND = "nccl"
INIT_METHOD = "tcp://localhost:54321"

flickr_boundaries_fn = Path('/storage/Datasets/flickr_audio/flickr_8k.ctm')
flickr_audio_dir = flickr_boundaries_fn.parent / "wavs"
flickr_images_fn = Path('/storage/Datasets/Flicker8k_Dataset/')
flickr_segs_fn = Path('./data/flickr_image_masks/')

config_library = {
    "multilingual": "English_Hindi_DAVEnet_config.json",
    "multilingual+matchmap": "English_Hindi_matchmap_DAVEnet_config.json",
    "english": "English_DAVEnet_config.json",
    "english+matchmap": "English_matchmap_DAVEnet_config.json",
    "hindi": "Hindi_DAVEnet_config.json",
    "hindi+matchmap": "Hindi_matchmap_DAVEnet_config.json",
}

scipy_windows = {
    'hamming': scipy.signal.hamming,
    'hann': scipy.signal.hann, 
    'blackman': scipy.signal.blackman,
    'bartlett': scipy.signal.bartlett
    }

def myRandomCrop(im, resize, to_tensor):

        im = resize(im)
        im = to_tensor(im)
        return im

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

    return torch.tensor(logspec), nsamples#, n_frames

def LoadImage(impath, resize, image_normalize, to_tensor):
    img = Image.open(impath).convert('RGB')
    # img = self.image_resize_and_crop(img)
    img = myRandomCrop(img, resize, to_tensor)
    img = image_normalize(img)
    return img

def PadFeat(feat, target_length, padval):
    nframes = feat.shape[1]
    pad = target_length - nframes

    if pad > 0:
        feat = np.pad(feat, ((0, 0), (0, pad)), 'constant',
            constant_values=(padval, padval))
    elif pad < 0:
        nframes = target_length
        feat = feat[:, 0: pad]

    return torch.tensor(feat).unsqueeze(0), torch.tensor(nframes).unsqueeze(0)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--resume", action="store_true", dest="resume",
        help="load from exp_dir if True")
parser.add_argument("--config-file", type=str, default='matchmap', choices=['matchmap'], help="Model config file.")
parser.add_argument("--restore-epoch", type=int, default=-1, help="Epoch to generate accuracies for.")
parser.add_argument("--image-base", default="..", help="Model config file.")
command_line_args = parser.parse_args()
restore_epoch = command_line_args.restore_epoch

# Setting up model specifics
heading(f'\nSetting up model files ')
args, image_base = modelSetup(command_line_args, True)
rank = 'cuda'
 
concepts = []
with open('./data/test_keywords.txt', 'r') as f:
    for keyword in f:
        concepts.append(keyword.strip())

aud_files = Path("../../Datasets/yfacc_v6")
# alignments = {}
# prev = ''
# prev_wav = ''
# prev_start = 0
# with open(aud_files / 'flickr_8k.ctm', 'r') as f:
#     for line in f:
#         name, _, start, dur, label = line.strip().split()
#         wav = name.split('.')[0] + '_' + name.split('#')[-1]
#         label = label.lower()
#         if label in concepts:
#             if wav not in alignments: alignments[wav] = {}
#             if label not in alignments[wav]: alignments[wav][label] = (int(float(start)*100), int((float(start) + float(dur))*100))
#         prev = label
#         prev_wav = wav
#         prev_start = start

yoruba_alignments = {}
translation = {}
yoruba_vocab = []
with open(Path('../../Datasets/yfacc_v6/Flickr8k_text/eng_yoruba_keywords.txt'), 'r') as f:
    for line in f:
        e, y = line.strip().split(', ')
        if e in concepts:
            translation[y] = e
            yoruba_vocab.append(y)
for y in translation:
    print(y, translation[y])

for txt_grid in Path('../../Datasets/yfacc_v6/Flickr8k_alignment').rglob('*.TextGrid'):
    if str(txt_grid) == '../../Datasets/yfacc_v6/Flickr8k_alignment/3187395715_f2940c2b72_0.TextGrid': continue
    grid = textgrids.TextGrid(txt_grid)
    wav = 'S001_' + txt_grid.stem
    
    for interval in grid['words']:
        
        x = str(interval).split()
        label = str(interval).split('"')[1]
        start = x[-2].split('=')[-1]
        dur = x[-1].split('=')[-1].split('>')[0]

        if label in yoruba_vocab:
            if wav not in yoruba_alignments: yoruba_alignments[wav] = {}
            if label not in yoruba_alignments[wav]: 
                yoruba_alignments[wav][label] = (int(float(start)*100), int((float(start) + float(dur))*100))

audio_conf = args["audio_config"]
target_length = audio_conf.get('target_length', 1024)
padval = audio_conf.get('padval', 0)
image_conf = args["image_config"]
crop_size = image_conf.get('crop_size')
center_crop = image_conf.get('center_crop')
RGB_mean = image_conf.get('RGB_mean')
RGB_std = image_conf.get('RGB_std')

# image_resize_and_crop = transforms.Compose(
#         [transforms.Resize(224), transforms.ToTensor()])
resize = transforms.Resize((256, 256))
to_tensor = transforms.ToTensor()
image_normalize = transforms.Normalize(mean=RGB_mean, std=RGB_std)

image_resize = transforms.transforms.Resize((256, 256))
trans = transforms.ToPILImage()

# Create models
audio_model = mutlimodal(args).to(rank)

seed_model = alexnet(pretrained=True)
image_model = nn.Sequential(*list(seed_model.features.children()))

last_layer_index = len(list(image_model.children()))
image_model.add_module(str(last_layer_index),
    nn.Conv2d(256, args["audio_model"]["embedding_dim"], kernel_size=(3,3), stride=(1,1), padding=(1,1)))
image_model = image_model.to(rank)

attention = ScoringAttentionModule(args).to(rank)
contrastive_loss = ContrastiveLoss(args).to(rank)

model_with_params_to_update = {
    "audio_model": audio_model,
    "attention": attention,
    "contrastive_loss": contrastive_loss,
    "image_model": image_model
    }
model_to_freeze = {
    }
trainable_parameters = getParameters(model_with_params_to_update, model_to_freeze, args)

if args["optimizer"] == 'sgd':
    optimizer = torch.optim.SGD(
        trainable_parameters, args["learning_rate_scheduler"]["initial_learning_rate"],
        momentum=args["momentum"], weight_decay=args["weight_decay"]
        )
elif args["optimizer"] == 'adam':
    optimizer = torch.optim.Adam(
        trainable_parameters, args["learning_rate_scheduler"]["initial_learning_rate"],
        weight_decay=args["weight_decay"]
        )
else:
    raise ValueError('Optimizer %s is not supported' % args["optimizer"])

scaler = torch.cuda.amp.GradScaler()

audio_model = DDP(audio_model, device_ids=[rank])
image_model = DDP(image_model, device_ids=[rank])
attention = DDP(attention, device_ids=[rank])

if "restore_epoch" in args:
    info, start_epoch, global_step, best_epoch, best_acc = loadModelAttriburesAndTrainingAtEpochAMP(
        args["exp_dir"], audio_model, image_model, attention, contrastive_loss, optimizer, rank, 
        args["restore_epoch"]
        )
else: 
    heading(f'\nRetoring model parameters from best epoch ')
    info, epoch, global_step, best_epoch, best_acc = loadModelAttriburesAndTrainingAMP(
        args["exp_dir"], audio_model, image_model, attention, contrastive_loss, optimizer, 
        rank, False
        )

audio_model.eval()
image_model.eval()
attention.eval()
contrastive_loss.eval()
image_base = Path('../../Datasets/Flicker8k_Dataset')
episodes = np.load(Path('./data/yoruba_test_episodes.npz'), allow_pickle=True)['episodes'].item()

with torch.no_grad():
    acc = []
    for i_test in range(1):
        print(f'\nTest number {i_test+1}-----------------------------------')
        results = {}
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        queries = {}
        query_names = {}

        episode_names = list(episodes.keys())

        # episode_names = np.random.choice(episode_names, 100, replace=False)

        for episode_num in tqdm(sorted(episode_names)):

            episode = episodes[episode_num]
            
            m_images = []
            m_labels = []
            counting = {}

            for w in episode['matching_set']:

                imgpath = image_base / Path(str(episode['matching_set'][w]) + '.jpg')
                this_image = LoadImage(imgpath, resize, image_normalize, to_tensor)
                this_image_output = image_model(this_image.unsqueeze(0).to(rank))
                this_image_output = this_image_output.view(this_image_output.size(0), this_image_output.size(1), -1).transpose(1, 2)
                # this_image_output = this_image_output.mean(dim=1)
                m_images.append(this_image_output)
                m_labels.append(w)

            for w in list(episode['matching_set']):
                if w not in concepts: continue
                if w not in counting: counting[w] = 0
                counting[w] += 1
                # if counting[w] == 10: break

            m_images = torch.cat(m_images, axis=0)
    
            for w in episode['queries']:
                if w in ['fire hydrant', 'kite']: continue
                if w not in results: results[w] = {'correct': 0, 'total': 0}
                wav = episode['queries'][w]

                lookup = str(Path(wav).stem)
                if lookup in yoruba_alignments:
                    if w in yoruba_alignments[lookup]:
                        this_english_audio_feat, this_english_nframes = LoadAudio(aud_files / Path('flickr_audio_yoruba_test') / Path(wav + '.wav'), yoruba_alignments[lookup][w], audio_conf)
                        this_english_audio_feat, this_english_nframes = PadFeat(this_english_audio_feat, target_length, padval)
                        _, _, query = audio_model(this_english_audio_feat.to(rank))
                        n_frames = NFrames(this_english_audio_feat, query, this_english_nframes) 
                        scores = attention.module.one_to_many_score(m_images, query, n_frames).squeeze()

                        ind = torch.argmax(scores).item()
                        if w == m_labels[ind]: 
                            results[w]['correct'] += 1
                        results[w]['total'] += 1
            # if episode_num == 99: break
            
            
        c = 0
        t = 0
        for w in results:
            correct = results[w]['correct']
            total = results[w]['total']
            c += correct
            t += total
            percentage = 100*correct/total
            print(f'{w}: {correct}/{total}={percentage:<.2f}%')
        percentage = c/t
        print(f'Overall: {c}/{t}={percentage}={100*percentage:<.2f}%')

        acc.append(100*percentage)

    avg = np.mean(np.asarray(acc))
    var = np.std(np.asarray(acc))
    print(f'\nOverall mean {avg}% and std {var}%')