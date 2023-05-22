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
import textgrids

num_episodes = 1000

aud_files = Path("../../../Datasets/yfacc_v6")
ss_save_fn = '../support_set/support_set.npz'
image_base = Path('../../../Datasets/Flicker8k_Dataset')
support_set = np.load(ss_save_fn, allow_pickle=True)['support_set'].item()
val_fn = Path('../data/test.json')
val = {}

vocab = []
with open('../data/test_keywords.txt', 'r') as f:
    for keyword in f:
        vocab.append(' '.join(keyword.split()))

yoruba_alignments = {}
translation = {}
y2etranslation = {}
yoruba_vocab = []
with open(Path('../../../Datasets/yfacc_v6/Flickr8k_text/eng_yoruba_keywords.txt'), 'r') as f:
    for line in f:
        e, y = line.strip().split(', ')
        if e in vocab:
            translation[e] = y.lower()
            y2etranslation[y.lower()] = e
            yoruba_vocab.append(y.lower())
for e in translation:
    print(e, translation[e])

for txt_grid in Path('../../../Datasets/yfacc_v6/Flickr8k_alignment').rglob('*.TextGrid'):
    if str(txt_grid) == '../../../Datasets/yfacc_v6/Flickr8k_alignment/3187395715_f2940c2b72_0.TextGrid': continue
    grid = textgrids.TextGrid(txt_grid)
    wav = txt_grid.stem
    
    for interval in grid['words']:
        
        x = str(interval).split()
        label = str(interval).split('"')[1]
        start = x[-2].split('=')[-1]
        dur = x[-1].split('=')[-1].split('>')[0]

        if label in yoruba_vocab:
            if wav not in yoruba_alignments: yoruba_alignments[wav] = {}
            if label not in yoruba_alignments[wav]: yoruba_alignments[wav][label] = (float(start), float(start)+float(dur))

captions = {}
for cap_fn in Path('../../../Datasets/yfacc_v6/Flickr8k_text').rglob('Flickr8k.token.*.txt'):
    with open(cap_fn, 'r') as f:
        for line in f:
            parts = line.split()
            cap = ' '.join(parts[1:]) + ' '
            name = parts[0].split('.')[0] + '_' + parts[0].split('#')[-1]
            captions[name] = cap

s_imgs = []
s_wavs = []
s_names = []

for wav_name in tqdm(support_set):

    eng, img, wav, start, dur, word, _, _, _ = support_set[wav_name]
    s_wavs.append(Path(wav).stem)
    s_imgs.append(Path(img).stem)
    s_names.append(Path(eng).stem)

with open(val_fn, 'r') as f:
    val = json.load(f)

special_characters = {
    'kẹ̀kẹ́': 'kẹ̀kẹ́',
    'ọmọ': 'Ọmọ'
}

data = []
words2aud = {}
for e_w in vocab:
    w = translation[e_w]
    for im, eng, yor in val[e_w]:
        name = Path(eng).stem
        if name in yoruba_alignments and name not in s_names:
            if w in yoruba_alignments[name]: 
                if re.search(w + ' ', captions[name]) is not None:
                    # print(w, captions[name])
                    data.append((im, eng, yor, w))
                    if w not in words2aud: words2aud[w] = []
                    words2aud[w].append(Path(yor).stem)

                elif w in special_characters:
                    if re.search(special_characters[w] + ' ', captions[name]) is not None:
                        data.append((im, eng, yor, w))
                        if w not in words2aud: words2aud[w] = []
                        words2aud[w].append(Path(yor).stem)

test_episodes = {}

##################################
# Test queries  
##################################

images = np.load(Path('../data/test_episodes_images.npz'), allow_pickle=True)['images'].item()
image_words = np.load(Path('../data/test_episodes_images.npz'), allow_pickle=True)['image_words'].item()

for word in yoruba_vocab:

    aud_instances = np.random.choice(np.arange(0, len(words2aud[word])), num_episodes)
    im_instances = np.random.choice(np.arange(0, len(images[y2etranslation[word]])), num_episodes)        
    for episode_num in tqdm(range(num_episodes)):

        if episode_num not in test_episodes: test_episodes[episode_num] = {'queries': {}, 'matching_set': {}, 'possible_words': {}}
        test_episodes[episode_num]['queries'][word] = (words2aud[word][aud_instances[episode_num]])
        test_episodes[episode_num]['matching_set'][word] = (images[y2etranslation[word]][im_instances[episode_num]])
        y_words = [translation[e] for e in image_words[images[y2etranslation[word]][im_instances[episode_num]]]]
        test_episodes[episode_num]['possible_words'][word] = y_words


for episode_n in range(num_episodes):
    if len(test_episodes[episode_num]['queries']) != len(yoruba_vocab) or len(test_episodes[episode_num]['matching_set']) != len(yoruba_vocab):
        print("BUG", set(test_episodes[episode_num]['queries'].keys()), set(yoruba_vocab))
        
test_save_fn = '../data/yoruba_test_episodes'
np.savez_compressed(
    Path(test_save_fn).absolute(), 
    episodes=test_episodes
    )