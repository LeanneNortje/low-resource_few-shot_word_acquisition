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

aud_files = Path("../../Datasets/flickr_audio")
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
words2imgs = {}
words2aud = {}
for w in val:
    for im, eng, yor in val[w]:
        data.append((im, eng, yor, w))
        if w not in words2imgs: words2imgs[w] = []
        words2imgs[w].append(Path(im).stem)
        if w not in words2aud: words2aud[w] = []
        words2aud[w].append(Path(eng).stem)

# images = {}
# audio = {}
# captions = {}
# with open(Path("../../Datasets/flickr_audio") / 'flickr_8k.ctm', 'r') as f:
#     for line in f:
#         name, _, start, dur, label = line.strip().split()
#         wav = name.split('.')[0] + '_' + name.split('#')[-1]
#         if wav not in captions: captions[wav] = label.lower()
#         else: 
#             captions[wav] += ' '
#             captions[wav] += label.lower()

# image_captions = {}
# for wav in captions:
#     caption = captions[wav]
#     name = '_'.join(wav.split('_')[0:2])
#     for v in vocab:
#         if re.search(v, caption) is not None:
#             if name not in image_captions: image_captions[name] = []
#             image_captions[name].append(captions[wav])

# audio_captions = {}
# for name in captions:
#     caption = captions[wav]
#     for v in vocab:
#         if re.search(v, caption) is not None:
#             if name not in audio_captions: audio_captions[name] = []
#             audio_captions[name].append(captions[wav])

# for name in tqdm(captions):
#     caption = captions[name]
#     c = False
#     for v in vocab:
#         if re.search(v, caption) is not None:
#             c = True
#     if c is False: 
#         neg_wavs.add(name)

# for entry in data: 
#     im = entry['image']
#     id = int(Path(im).stem.split('_')[-1])
#     for caption in entry['captions']:
#         for word in vocab:
#             if re.search(word, caption['text'].lower()) is not None and Path(caption['wav']).stem in alignments:# and word in image_annotations[id]:
#                 if word not in val: val[word] = []
#                 val[word].append((im, caption['wav'], caption['speaker'], id_to_caption[id]))

test_episodes = {}

# matching_set = {}

# ##################################
# # Test matching set 
# ##################################

# for entry in data:
#     im = entry['image']
#     if im not in matching_set: matching_set[im] = set()
#     for caption in entry['captions']:
        
#         for word in vocab:
#             if re.search(word, caption['text'].lower()) is not None:
#                 if im not in matching_set: matching_set[im] = set()
#                 matching_set[im].add(word)
#         # used_images.add(im)
# test_episodes['matching_set'] = matching_set
# print(len(matching_set))

##################################
# Test queries  
##################################

for word in vocab:

    aud_instances = np.random.choice(np.arange(0, len(words2aud[word])), num_episodes)
    im_instances = np.random.choice(np.arange(0, len(words2imgs[word])), num_episodes)        
    for episode_num in tqdm(range(num_episodes)):

        if episode_num not in test_episodes: test_episodes[episode_num] = {'queries': {}, 'matching_set': {}}
        test_episodes[episode_num]['queries'][word] = (words2aud[word][aud_instances[episode_num]])
        test_episodes[episode_num]['matching_set'][word] = (words2imgs[word][im_instances[episode_num]])


for episode_n in range(num_episodes):
    if len(test_episodes[episode_num]['queries']) != len(vocab) or len(test_episodes[episode_num]['matching_set']) != len(vocab):
        print("BUG")
test_save_fn = '../data/test_episodes'
np.savez_compressed(
    Path(test_save_fn).absolute(), 
    episodes=test_episodes
    )