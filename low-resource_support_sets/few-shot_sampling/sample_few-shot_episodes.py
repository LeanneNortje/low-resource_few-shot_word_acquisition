#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

from pathlib import Path
import json
import re
import numpy as np
from tqdm import tqdm
import string

num_episodes = 1000

aud_files = Path("../../../Datasets/flickr_audio")
ss_save_fn = '../support_set/support_set.npz'
image_base = Path('../../Datasets/Flicker8k_Dataset')
support_set = np.load(ss_save_fn, allow_pickle=True)['support_set'].item()
val_fn = Path('../data/test.json')
val = {}

vocab = []
with open('../data/test_keywords.txt', 'r') as f:
    for keyword in f:
        vocab.append(' '.join(keyword.split()))

# annotations_fn = Path("../../Datasets/spokencoco/captions_val2017.json")
# f = json.load(open(annotations_fn))
# annotations = f['annotations']
# im_annotations = f['images']

# id_to_caption = {}
# for entry in annotations: 
#     if entry['image_id'] not in id_to_caption: id_to_caption[entry['image_id']] = []
#     id_to_caption[entry['image_id']].extend(entry['caption'].lower().translate(str.maketrans('', '', string.punctuation)).split())

# for id in id_to_caption: 
#     id_to_caption[id] = set(id_to_caption[id])

# for entry in im_annotations: 
#     id = int(entry['file_name'].split('.')[0])
#     print(id, entry['file_name'], id_to_caption[id])
#     break

alignments = {}
prev = ''
prev_wav = ''
prev_start = 0
with open(aud_files / 'flickr_8k.ctm', 'r') as f:
    for line in f:
        name, _, start, dur, label = line.strip().split()
        wav = name.split('.')[0] + '_' + name.split('#')[-1]
        label = label.lower()
        if label in vocab:
            if wav not in alignments: alignments[wav] = {}
            if label not in alignments[wav]: alignments[wav][label] = (float(start), float(start) + float(dur))
        prev = label
        prev_wav = wav
        prev_start = start

s_imgs = []
s_wavs = []

for wav_name in tqdm(support_set):

    wav, img, yor, start, dur, word, _, _, _ = support_set[wav_name]
    s_wavs.append(Path(wav).stem)
    s_imgs.append(Path(img).stem)

with open(val_fn, 'r') as f:
    val = json.load(f)

data = []
words2aud = {}
for w in val:
    for im, eng, yor in val[w]:
        data.append((im, eng, yor, w))
        if w not in words2aud: words2aud[w] = []
        words2aud[w].append(Path(eng).stem)

test_episodes = {}

##################################
# Test queries  
##################################

images = np.load(Path('../data/test_episodes_images.npz'), allow_pickle=True)['images'].item()


for word in vocab:

    aud_instances = np.random.choice(np.arange(0, len(words2aud[word])), num_episodes)
    im_instances = np.random.choice(np.arange(0, len(images[word])), num_episodes)        
    for episode_num in tqdm(range(num_episodes)):

        if episode_num not in test_episodes: test_episodes[episode_num] = {'queries': {}, 'matching_set': {}}
        test_episodes[episode_num]['queries'][word] = (words2aud[word][aud_instances[episode_num]])
        test_episodes[episode_num]['matching_set'][word] = (images[word][im_instances[episode_num]])


for episode_n in range(num_episodes):
    if len(test_episodes[episode_num]['queries']) != len(vocab) or len(test_episodes[episode_num]['matching_set']) != len(vocab):
        print("BUG")
test_save_fn = '../data/test_episodes'
np.savez_compressed(
    Path(test_save_fn).absolute(), 
    episodes=test_episodes
    )