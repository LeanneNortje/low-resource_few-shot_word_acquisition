#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

from pathlib import Path
import json
import re
from tqdm import tqdm
import numpy as np

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize, sent_tokenize

aud_files = Path("../../Datasets/flickr_audio")
im_files = Path("../../Datasets/Flicker8k_Dataset")
yoruba_files = Path("../../Datasets/yfacc_v6")
##########################
concepts = [] 
with open('./data/test_keywords.txt', 'r') as f:
    for keyword in f:
        concepts.append(' '.join(keyword.split()))

# word_to_name = {}
name_to_word = {}
with open(aud_files / 'flickr_8k.ctm', 'r') as f:
    for line in f:
        word = line.split()[-1].lower()
        if word in concepts:
            # if word not in word_to_name: word_to_name[word] = []
            name = line.split()[0].split('.')[0] + '_' + line.split()[0].split('#')[-1]
            # word_to_name[word].append(name)

            if name not in name_to_word: name_to_word[name] = []
            name_to_word[name].append(word)
        
##########################
rest = {}
# train_fn = Path(aud_files) / 'SpokenCOCO_train.json'
# val_fn = Path(files) / 'SpokenCOCO_val.json'

splits = {}
for y_wav in yoruba_files.rglob('*.wav'):
    split = y_wav.parent.stem.split('_')[-1]
    if split not in splits: splits[split] ={}
    name = '_'.join(y_wav.stem.split('_')[1:3])
    if name not in splits[split]: splits[split][name] = {'english': [], 'yoruba': [], 'images': []}
    splits[split][name]['yoruba'].append(y_wav)


for e_wav in aud_files.rglob('*.wav'):
    name = '_'.join(e_wav.stem.split('_')[0:2])
    added = False
    for s in splits:
        if name in splits[s]: 
            splits[s][name]['english'].append(e_wav)
            added = True
    if added is False: 
        if name not in rest: rest[name] = {'english': [], 'images': []}
        rest[name]['english'].append(e_wav)

for im_fn in im_files.rglob('*.jpg'):
    name = im_fn.stem
    added = False
    for s in splits:
        if name in splits[s]: 
            splits[s][name]['images'].append(im_fn)
            added = True
    if added is False: 
        if name not in rest: rest[name] = {'english': [], 'images': []}
        rest[name]['images'].append(im_fn)

test = {}
val = {}
train = {}
for base_name in splits['test']:
    
    for yor_wav in splits['test'][base_name]['yoruba']:
        name = '_'.join(yor_wav.stem.split('_')[1:])
        if name not in name_to_word: continue
        for w in name_to_word[name]:
            if w not in test: test[w] = []
            image = splits['test'][base_name]['images'][0]
            # yor_wav = splits['test'][base_name]['yoruba'][0]
            for e_wav in splits['test'][base_name]['english']:
                if e_wav.stem == name:
                    wav = e_wav
                    break
            test[w].append((str(image), str(wav), str(yor_wav)))

for base_name in splits['dev']:

    for yor_wav in splits['dev'][base_name]['yoruba']:
        name = '_'.join(yor_wav.stem.split('_')[1:])
        if name not in name_to_word: continue
        for w in name_to_word[name]:
            if w not in val: val[w] = []
            image = splits['dev'][base_name]['images'][0]
            for e_wav in splits['dev'][base_name]['english']:
                if e_wav.stem == name:
                    wav = e_wav
                    break
            val[w].append((str(image), str(wav), str(yor_wav)))

for base_name in splits['train']:

    for yor_wav in splits['train'][base_name]['yoruba']:
        name = '_'.join(yor_wav.stem.split('_')[1:])
        if name not in name_to_word: continue
        for w in name_to_word[name]:
            if w not in train: train[w] = []
            image = splits['train'][base_name]['images'][0]
            for e_wav in splits['train'][base_name]['english']:
                if e_wav.stem == name:
                    wav = e_wav
                    break
            train[w].append((str(image), str(wav), str(yor_wav)))


sorting = {}
for w in train:
    e_1 = len(train[w]) + len(val[w])
    e_2 = len(test[w])
    # print(f'{w}\t{e_1}:{e_2}')
    sorting[w] = e_1 + e_2

for entry in sorted(sorting.items(), key=lambda x:x[1], reverse=True):
    w = entry[0]
    e_1 = len(train[w]) + len(val[w])
    e_2 = len(test[w])
    print(f'{w}\t{e_1}:{e_2}')

fn = Path('./data/train.json')
with open(fn, 'w') as f:
    json.dump(train, f)
print(f'Num train classes: {len(train)}')

fn = Path('./data/val.json')
with open(fn, 'w') as f:
    json.dump(val, f)
print(f'Num val classes: {len(val)}')

fn = Path('./data/test.json')
with open(fn, 'w') as f:
    json.dump(test, f)
print(f'Num val classes: {len(test)}')

key = {}
id_to_word_key = {}
for i, l in enumerate(concepts):
    key[l] = i
    id_to_word_key[i] = l
    print(f'{i}: {l}')

np.savez_compressed(
    Path('data/label_key.npz'),
    key=key,
    id_to_word_key=id_to_word_key
)