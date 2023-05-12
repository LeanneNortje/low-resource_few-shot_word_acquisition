#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

from tqdm import tqdm
from pathlib import Path
import json
import re
import numpy as np
import torchaudio
from IPython.display import Audio, display
import textgrids

K = 5

# def load_fn(fn, word_points, vocab):
#     count = {}
#     with open(fn, 'r') as f:
#         data = json.load(f)

#     data = data['data']

#     for entry in tqdm(data):
#         image = entry['image']

#         for caption in entry['captions']:
            
#             for word in vocab:
#                 if re.search(word, caption['text'].lower()) is not None:
#                     if word not in word_points:
#                         word_points[word] = []
#                     word_points[word].append((image, caption['wav'], caption['speaker']))
#                     if word not in count: count[word] = 0
#                     count[word] += 1

#     print('Word counts')
#     for word in count:
#         print(f'{word}: {count[word]}')
                    

aud_files = Path("../../Datasets/flickr_audio")
save_dir = Path('../support_set')
save_dir.mkdir(parents=True, exist_ok=True)

vocab = []
with open('../data/test_keywords.txt', 'r') as f:
    for keyword in f:
        vocab.append(' '.join(keyword.split()))

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

yoruba_alignments = {}
translation = {}
yoruba_vocab = []
with open(Path('../../Datasets/yfacc_v6/Flickr8k_text/eng_yoruba_keywords.txt'), 'r') as f:
    for line in f:
        e, y = line.strip().split(', ')
        if e in vocab:
            translation[e] = y
            yoruba_vocab.append(y)
for e in translation:
    print(e, translation[e])

for txt_grid in Path('../../Datasets/yfacc_v6/Flickr8k_alignment').rglob('*.TextGrid'):
    if str(txt_grid) == '../../Datasets/yfacc_v6/Flickr8k_alignment/3187395715_f2940c2b72_0.TextGrid': continue
    grid = textgrids.TextGrid(txt_grid)
    wav = txt_grid.stem
    
    for interval in grid['words']:
        
        x = str(interval).split()
        label = str(interval).split('"')[1]
        start = x[-2].split('=')[-1]
        dur = x[-1].split('=')[-1].split('>')[0]

        if label in yoruba_vocab:
            print(label)
            if wav not in yoruba_alignments: yoruba_alignments[wav] = {}
            if label not in yoruba_alignments[wav]: yoruba_alignments[wav][label] = (float(start), float(start)+float(dur))


support_set = {}

##################################
# Support set 
##################################
fn = Path('../data/test.json')
with open(fn, 'r') as f:
    train = json.load(f)
word_counts = {}
word_names = {}

if Path(save_dir / Path('support_set.npz')).is_file():
    support_set = np.load(Path(save_dir / Path('support_set.npz')), allow_pickle=True)['support_set'].item()
    for name in support_set.copy():

        entry = support_set[name]
        word = entry[-2]
        if word not in word_counts: word_counts[word] = 0
        word_counts[word] += 1
        if word not in word_names: word_names[word] = []
        if name in word_names[word]: support_set.pop(name)
        else: word_names[word].append(name)
        if word_counts[word] > K: support_set.pop(name)

for word in vocab:
    # filtered = [(i, w, s) for i, w, s in train[word] if Path(w).stem in alignments and Path(w).stem not in support_set]
    y_word = translation[word]
    filtered = []
    images = []
    for i, e, y in train[word]:

        if Path(e).stem in alignments and Path(e).stem not in support_set:
            if i not in images: 
                filtered.append((i, e, y))
                images.append(i)
    instances = np.arange(0, len(filtered))
    np.random.shuffle(instances)

    count = 0
    if word in word_counts:
        count = word_counts[word]
    
    for im, wav, yor in [filtered[i] for i in instances]:
        name = Path(wav).stem
        if name in support_set or count == K or name not in yoruba_alignments: continue
        dur = int(((float(alignments[name][word][1])-float(alignments[name][word][0])))*16000)
        offset = int(float(alignments[name][word][0])*16000)
        print(offset, dur, alignments[name][word])
        aud, sr = torchaudio.load(Path('..') / wav, frame_offset=offset, num_frames=dur)
        torchaudio.save(Path('..') / Path('temp.wav'), aud, sr)
        ans = input(f'{count} / {K} {word}({y_word}): ')
        if ans == 'y':
            
            save_name = Path(wav).stem  + '_' + word + '.wav'
            out_path = save_dir / Path(wav).parent.stem / Path(save_name)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(out_path, aud, sr)

            y_word = translation[word]
            y_dur = int(((float(yoruba_alignments[name][y_word][1])-float(yoruba_alignments[name][y_word][0]))*0.02 )*48000)
            y_offset = int(float(yoruba_alignments[name][y_word][0])*48000*0.02)
            aud, sr = torchaudio.load(Path('..') / yor, frame_offset=y_offset, num_frames=y_dur)

            save_name = Path(yor).stem  + '_' + y_word + '.wav'
            out_path = save_dir / Path(wav).parent / Path(save_name)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(out_path, aud, sr)

            support_set[name] = (wav, yor, im, alignments[name][word][0], alignments[name][word][1], word, y_word, yoruba_alignments[name][y_word][0], yoruba_alignments[name][y_word][1])

            np.savez_compressed(
                save_dir / Path('support_set'), 
                support_set=support_set
                )
            count += 1
        if count == K: break

np.savez_compressed(
    save_dir / Path('support_set'), 
    support_set=support_set
    )