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
image_base = Path('../../../Datasets/Flicker8k_Dataset')
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

embeddings = np.load(Path("../data/image_embeddings.npz"), allow_pickle=True)["embeddings"].item()
s_fns = []
s_images = {}
s_words = []
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

for wav_name in tqdm(support_set):
    wav, img, yor, start, dur, word, _, _, _ = support_set[wav_name]
    name = Path(img).stem
    image = LoadImage(Path('../..') / img, resize, image_normalize, to_tensor).unsqueeze(0)
    im = image_model(image.to(rank))

    if word not in s_images: s_images[word] = []
    s_images[word].append(im)
    s_words.append(word)
    s_fns.append(img)

for word in s_images:
    s_images[word] = torch.cat(s_images[word], dim=0)

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

query_scores = {}

for q_word in s_images:
    id = key[q_word]
    q_images = s_images[q_word]
    
    for entry in tqdm(data, desc=f'{q_word}({id})'):
        image = entry[0]
        if image in s_fns: continue
        name = Path(image).stem
        image_output = torch.tensor(embeddings[name]).unsqueeze(0).to(rank)

        sim = cos(image_output, q_images)
        ind = torch.argsort(sim, descending=True)

        if id not in query_scores: query_scores[id] = {'values': [], 'imgs': []}
        if image not in query_scores[id]['imgs']:
            query_scores[id]['values'].append(sim[ind[0]].item())
            query_scores[id]['imgs'].append(image)  
            

for id in query_scores:
    print(len(query_scores[id]['values']), len(query_scores[id]['imgs']))

newly_labeled = {}
top_N = 100

for id in query_scores:
    
    indices = np.argsort(query_scores[id]['values'])[::-1]

    for i in range(top_N):
        img =  Path(query_scores[id]['imgs'][indices[i]])
        img_name = img.stem

        if id not in newly_labeled: newly_labeled[id] = []
        newly_labeled[id].append(Path(*img.parts[1:]))

np.savez_compressed(
    Path("../data/sampled_img_data"), 
    data=newly_labeled
)