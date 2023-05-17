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

vocab = []
with open('./data/test_keywords.txt', 'r') as f:
    for keyword in f:
        vocab.append(' '.join(keyword.split()))

key = np.load(Path('./data/label_key.npz'), allow_pickle=True)['key'].item()
id_to_word_key = np.load(Path('./data/label_key.npz'), allow_pickle=True)['id_to_word_key'].item()
for l in key:
    print(f'{key[l]}: {l}')

data = np.load(Path("./data/sampled_audio_data.npz"), allow_pickle=True)['data'].item()

audio_list_dir = Path('./data/audio_pair_lists')
audio_list_dir.mkdir(parents=True, exist_ok=True)

for id in data:
    fn = audio_list_dir / f'{id_to_word_key[id]}.txt'
    # print(id, id_to_word_key[id])
    with open(fn, 'w') as f:
        for wav in data[id]:
            wav_name = wav.stem
            f.write(f'{wav_name}\n')

data = np.load(Path("./data/sampled_img_data.npz"), allow_pickle=True)['data'].item()

image_list_dir = Path('./data/image_pair_lists')
image_list_dir.mkdir(parents=True, exist_ok=True)

for id in data:
    fn = image_list_dir / f'{id_to_word_key[id]}.txt'
    # print(id, id_to_word_key[id])
    with open(fn, 'w') as f:
        for im in data[id]:
            im_name = im.stem
            f.write(f'{im_name}\n')