#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import torch
from torchvision.io import read_image
from torchvision.models import *
from torchvision import transforms
from PIL import Image

BACKEND = "nccl"
INIT_METHOD = "tcp://localhost:54321"

def LoadImage(impath, resize, image_normalize, to_tensor):

    img = Image.open(impath).convert('RGB')
    img = resize(img)
    img = to_tensor(img)
    img = image_normalize(img)
    return img

ss_save_fn = '../support_set/support_set.npz'
image_base = Path('../....//Datasets/Flicker8k_Dataset')
support_set = np.load(ss_save_fn, allow_pickle=True)['support_set'].item()
files = "../../../Datasets/flickr_audio"
train_fn = Path('../../../Datasets/yfacc_v6/Flickr8k_text/Flickr8k.token.train_yoruba.txt')
val_fn = Path('../../../Datasets/yfacc_v6/Flickr8k_text/Flickr8k.token.dev_yoruba.txt')

vocab = []
with open('../data/test_keywords.txt', 'r') as f:
    for keyword in f:
        vocab.append(' '.join(keyword.split()))

key = np.load(Path('../data/label_key.npz'), allow_pickle=True)['key'].item()
id_to_word_key = np.load(Path('../data/label_key.npz'), allow_pickle=True)['id_to_word_key'].item()
for l in key:
    print(f'{key[l]}: {l}')

image_conf = {
    "crop_size": 224,
    "center_crop": False,
    "RGB_mean": [0.485, 0.456, 0.406],
    "RGB_std": [0.229, 0.224, 0.225]
}
RGB_mean = image_conf.get('RGB_mean')
RGB_std = image_conf.get('RGB_std')
resize = transforms.Resize((256, 256))
to_tensor = transforms.ToTensor()
image_normalize = transforms.Normalize(mean=RGB_mean, std=RGB_std)

rank = 'cuda'
image_model = alexnet(pretrained=True).to(rank)
image_model.eval()

image_base = Path('../../Datasets/Flicker8k_Dataset')
english_base = Path('../../Datasets/flickr_audio/wavs')
yoruba_base = Path('../../Datasets/yfacc_v6/flickr_audio_yoruba_train')

data = []
with open(train_fn, 'r') as f:
    for line in f:
        name = line.split()[0].split('.')[0]
        num = line.split()[0].split('#')[-1]
        im = image_base / Path(name + '.jpg')
        eng = english_base / Path(name + '_' + num + '.wav')
        yor = yoruba_base / Path('S001_' + name + '_' + num + '.wav')
        data.append((im, eng, yor))

with open(val_fn, 'r') as f:
    for line in f:
        name = line.split()[0].split('.')[0]
        num = line.split()[0].split('#')[-1]
        im = image_base / Path(name + '.jpg')
        eng = english_base / Path(name + '_' + num + '.wav')
        yor = yoruba_base / Path('S001_' + name + '_' + num + '.wav')
        data.append((im, eng, yor))

image_embed = {}
image_names = []
images = []
    
for c, (im, eng, yor) in tqdm(enumerate(data)):

    images.append(LoadImage(Path('..') / im, resize, image_normalize, to_tensor).unsqueeze(0))
    image_names.append(Path(im).stem)

    if len(images) == 128 or c == len(data)-1:

        images = torch.cat(images, dim=0)
    
        image_output = image_model(images.to(rank))

        for i, name in enumerate(image_names):
            image_embed[name] = image_output[i, :].cpu().detach().numpy()
        images = []
        image_names = []

np.savez_compressed(
    Path("../data/image_embeddings"), 
    embeddings=image_embed
)