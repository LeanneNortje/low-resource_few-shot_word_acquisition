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
import re

vocab = []
with open('../data/test_keywords.txt', 'r') as f:
    for keyword in f:
        vocab.append(' '.join(keyword.split()))

train_fn = Path('../data/train.json')
val_fn = Path('../data/val.json')

with open(train_fn, 'r') as f:
    train = json.load(f)

with open(val_fn, 'r') as f:
    val = json.load(f)

data = []
counts = {}
for w in train:
    if w not in counts: counts[w] = 0
    for im, eng, yor in train[w]:
        data.append((im, eng, yor, w))
        counts[w] += 1
        
for w in val:
    for im, eng, yor in val[w]:
        data.append((im, eng, yor, w))
        counts[w] += 1

neg_imgs = set()
neg_wavs = set()

captions = {}
with open(Path("../../../Datasets/flickr_audio") / 'flickr_8k.ctm', 'r') as f:
    for line in f:
        name, _, start, dur, label = line.strip().split()
        wav = name.split('.')[0] + '_' + name.split('#')[-1]
        if wav not in captions: captions[wav] = label.lower()
        else: 
            captions[wav] += ' '
            captions[wav] += label.lower()

image_captions = {}
for wav in captions:
    name = '_'.join(wav.split('_')[0:2])
    if name not in image_captions: image_captions[name] = []
    image_captions[name].append(captions[wav])

for name in tqdm(image_captions):

    for caption in image_captions[name]:
        c = False
        for v in vocab:
            if re.search(v, caption) is not None:
                c = True
        if c is False: 
            neg_imgs.add(name)

for name in tqdm(captions):
    caption = captions[name]
    c = False
    for v in vocab:
        if re.search(v, caption) is not None:
            c = True
    if c is False: 
        neg_wavs.add(name)


# for image, eng, yor, w in tqdm(data):

#     for caption in image_captions[Path(image).stem]:
#         c = False
#         for v in vocab:
#             if re.search(v, caption) is not None:
#                 c = True
#         if c is False: 
#             neg_imgs.add(image)
#             neg_wavs.add(wav)

val_neg_imgs = np.random.choice(list(neg_imgs), 100, replace=False)
train_neg_imgs = [entry for entry in list(neg_imgs) if entry not in val_neg_imgs]
val_neg_wavs = np.random.choice(list(neg_wavs), 100, replace=False)
train_neg_wavs = [entry for entry in list(neg_wavs) if entry not in val_neg_wavs]

key = np.load(Path('../data/label_key.npz'), allow_pickle=True)['key'].item()
id_to_word_key = np.load(Path('../data/label_key.npz'), allow_pickle=True)['id_to_word_key'].item()
for l in key:
    print(f'{key[l]}: {l}')

audio_samples = np.load(Path("../data/sampled_audio_data.npz"), allow_pickle=True)['data'].item()
image_samples = np.load(Path("../data/sampled_img_data.npz"), allow_pickle=True)['data'].item()

ss_save_fn = '../support_set/support_set.npz'
support_set = np.load(ss_save_fn, allow_pickle=True)['support_set'].item()

s = {}
for wav_name in support_set:
    wav, img, yor, start, dur, word, _, _, _= support_set[wav_name]
    if key[word] not in s: s[key[word]] = []
    s[key[word]].append(support_set[wav_name])
print(len(s))

val_number = 10
aud_train = {}
aud_val = {}

counts = {}
audio_segments_dir = Path('../../QbERT/segments_flickr_yoruba')

for id in audio_samples: 

    if id not in counts: counts[id] = {}
    for this_fn in audio_samples[id]:
        # this_fn = Path(wav).relative_to(*Path(wav).parts[:4]).with_suffix('.npz')
        # this_fn = audio_segments_dir / this_fn
        # for fn in this_fn.parent.rglob(f'*{this_fn.stem}.npz'):
        #     print(fn)
        # break
        name = this_fn.stem
        if name not in counts[id]: counts[id][name] = 0
        counts[id][name] += 1

image_counts = {}

for id in image_samples: 

    if id not in image_counts: image_counts[id] = {}
    for this_fn in image_samples[id]:
        name = this_fn.stem
        if name not in image_counts[id]: image_counts[id][name] = 0
        image_counts[id][name] += 1



keyword_count = {}
train_used = set()
val_used = set()
for id in counts:

    if id not in keyword_count: keyword_count[id] = 0

    for name in counts[id]:
        if counts[id][name] == 1 and keyword_count[id] < val_number and name not in train_used:
            if id not in aud_val: aud_val[id] = []
            aud_val[id].append(name)
            val_used.add(name)
            keyword_count[id] += 1
        elif name not in val_used:
            if id not in aud_train: aud_train[id] = []
            aud_train[id].append(name)
            train_used.add(name)
            keyword_count[id] += 1


train = {}
val = {}
# for id in image_samples:
#     # have_to_add = [str(img) for img in image_samples[id] if str(img) in val and str(img) not in train]
#     # options = [str(img) for img in image_samples[id] if str(img) not in val and str(img) not in train]
#     # v = np.random.choice(options, val_number-len(have_to_add), replace=False)
#     # val.extend(v)
#     # val.extend(have_to_add)
#     # # t = list(set(audio_samples[id]) - set(v))
#     # options = [str(img) for img in image_samples[id] if str(img) not in val]
#     # train.extend(options)
#     for name in aud_val[id]:
#         im_name = '_'.join(name.split('_')[0:2])
#         val.append(im_name)
#     for name in aud_train[id]:
#         im_name = '_'.join(name.split('_')[0:2])
#         train.append(im_name)

keyword_count = {}
train_used = set()
val_used = set()
for id in image_counts:

    if id not in keyword_count: keyword_count[id] = 0

    for name in image_counts[id]:
        if image_counts[id][name] == 1 and keyword_count[id] < val_number and name not in train_used:
            if id not in val: val[id] = []
            val[id].append(name)
            val_used.add(name)
            keyword_count[id] += 1
        elif name not in val_used:
            if id not in train: train[id] = []
            train[id].append(name)
            train_used.add(name)
            keyword_count[id] += 1

# Training 
audio_segments_dir = Path('../../QbERT/segments_flickr_yoruba')

def get_yoruba_wav(wav):
    # this_fn = Path(wav).relative_to(*Path(wav).parts[:4]).with_suffix('.npz')
    # for segment_fn in audio_segments_dir.rglob(f'*{this_fn.stem}.npz'):
    #     return segment_fn
    return wav

pos_lookup = {}

for id in s:
    if id not in pos_lookup: pos_lookup[id] = {"audio": {}, "images": {}}
    for wav, image, yor, start, dur, word, _, _, _ in s[id]:
        
        wav_name = Path(wav).stem
        if wav_name not in pos_lookup[id]['audio']: pos_lookup[id]['audio'][wav_name] = []
        pos_lookup[id]['audio'][wav_name].append(wav)

        image_name = Path(image).stem
        if image_name not in pos_lookup[id]['images']: pos_lookup[id]['images'][image_name] = []
        pos_lookup[id]['images'][image_name].append(Path('/'.join(str(image).split('/')[1:])))

for id in audio_samples:

    if id not in pos_lookup: pos_lookup[id] = {"audio": {}, "images": {}}
    for wav in audio_samples[id]:
        wav_name = wav.stem
        wav = str(get_yoruba_wav(wav))
        if wav_name in aud_train[id]:
            if wav_name not in pos_lookup[id]['audio']: pos_lookup[id]['audio'][wav_name] = []
            pos_lookup[id]['audio'][wav_name].append(wav)

neg_lookup = {}
for id in tqdm(sorted(pos_lookup), desc='Sampling audio training negatives'):
    wavs_with_id = list(pos_lookup[id]['audio'].keys())
    all_ids = list(pos_lookup.keys())
    all_ids.remove(id)

    if id not in neg_lookup: neg_lookup[id] = {"audio": {}, "images": {}}
    
    for neg_id in all_ids: #tqdm(all_ids, desc=f'ID: {id}'):
        temp = [i for name in pos_lookup[neg_id]['audio'] for i in pos_lookup[neg_id]['audio'][name] if name not in wavs_with_id]
        if len(temp) > 0:
            neg_lookup[id]['audio'][neg_id] = []
            # for w in list(set(train_neg_wavs) - set(temp)):
            for w in list(set(temp)):
                neg_lookup[id]['audio'][neg_id].append(w)


for id in image_samples:
    if id not in pos_lookup: continue #pos_lookup[id] = {"audio": {}, "images":{}, "neg_images": {}}
    for image in image_samples[id]:
        image_name = image.stem
        image = str(image)
        if image_name in train[id]: 
            if image_name not in pos_lookup[id]['images']: pos_lookup[id]['images'][image_name] = []
            pos_lookup[id]['images'][image_name].append(str(image))


print("*", len(train_neg_imgs))
for id in tqdm(sorted(pos_lookup), desc='Sampling image training negatives'):
    images_with_id = list(set([im for im in pos_lookup[id]['images']]))
    all_ids = list(pos_lookup.keys())
    all_ids.remove(id)

    if id not in neg_lookup: neg_lookup[id] = {"audio": {}, "images": {}}
        
    for neg_id in tqdm(all_ids, desc=f'ID: {id}'):
        temp = [i for name in pos_lookup[neg_id]['images'] for i in pos_lookup[neg_id]['images'][name] if name not in images_with_id]
        if len(temp) > 0:
            neg_lookup[id]['images'][neg_id] = temp 
            # neg_lookup[id]['images'][neg_id].extend(list(set(train_neg_imgs)-set(temp)))
            train_neg_imgs = list(set(train_neg_imgs) - set(temp))
print(len(train_neg_imgs))

for id in pos_lookup:
    print(id, " audio ", len(pos_lookup[id]['audio']), " images ", len(pos_lookup[id]['images']))

np.savez_compressed(
    Path("../data/train_lookup"), 
    lookup=pos_lookup,
    neg_lookup=neg_lookup,
    base_negs=train_neg_imgs
    )

# Validation
pos_lookup = {}

for id in audio_samples:

    if id not in pos_lookup: pos_lookup[id] = {"audio": {}, "images": {}}
    for wav in audio_samples[id]:
        wav_name = wav.stem
        wav = str(get_yoruba_wav(wav))

        if wav_name in aud_val[id]:
            if wav_name not in pos_lookup[id]['audio']: pos_lookup[id]['audio'][wav_name] = []
            pos_lookup[id]['audio'][wav_name].append(wav)

neg_lookup = {}
for id in tqdm(sorted(pos_lookup), desc='Sampling audio validation negatives'):
    wavs_with_id = list(pos_lookup[id]['audio'].keys())
    all_ids = list(pos_lookup.keys())
    all_ids.remove(id)

    if id not in neg_lookup: neg_lookup[id] = {"audio": {}, "images": {}}
    
    for neg_id in tqdm(all_ids, desc=f'ID: {id}'):
        temp = [i for name in pos_lookup[neg_id]['audio'] for i in pos_lookup[neg_id]['audio'][name] if name not in wavs_with_id]
        # if len(temp) > 0:
        neg_lookup[id]['audio'][neg_id] = temp
        # neg_lookup[id]['audio'][neg_id].extend(list(set(val_neg_wavs) - set(temp)))
        # for w in list(set(val_neg_wavs) - set(temp)):
        for w in list(set(temp)):
            neg_lookup[id]['audio'][neg_id].append(w)

for id in image_samples:
    if id not in pos_lookup: continue #pos_lookup[id] = {"audio": {}, "images":{}, "neg_images": {}}
    for image in image_samples[id]:
        image_name = image.stem
        image = str(image)
        if image_name in val[id]: 
            if image_name not in pos_lookup[id]['images']: pos_lookup[id]['images'][image_name] = []
            pos_lookup[id]['images'][image_name].append(str(image))  

for id in tqdm(sorted(pos_lookup), desc='Sampling image validation negatives'):
    images_with_id = list(set([im for im in pos_lookup[id]['images']]))
    all_ids = list(pos_lookup.keys())
    all_ids.remove(id)

    if id not in neg_lookup: neg_lookup[id] = {"audio": {}, "images": {}}
        
    for neg_id in tqdm(all_ids, desc=f'ID: {id}'):
        temp = [i for name in pos_lookup[neg_id]['images'] for i in pos_lookup[neg_id]['images'][name] if name not in images_with_id]
        # if len(temp) > 0:
        neg_lookup[id]['images'][neg_id] = temp 
        val_neg_imgs = list(set(val_neg_imgs) - set(temp))
        # neg_lookup[id]['images'][neg_id].extend(list(set(val_neg_imgs)-set(temp)))

for id in pos_lookup:
    print(id, " audio ", len(pos_lookup[id]['audio']), " images ", len(pos_lookup[id]['images']))

print()
remove = []
for id in neg_lookup:
    for neg_id in neg_lookup[id]['images']:
        if len(neg_lookup[id]['images'][neg_id]) == 0:
            remove.append((id, neg_id))

for id, neg_id in remove:
    del neg_lookup[id]['images'][neg_id]
for id in neg_lookup:
    print(id, " audio ", len(neg_lookup[id]['audio']), " images ", len(set(neg_lookup[id]['images'])))


np.savez_compressed(
    Path("../data/val_lookup"), 
    lookup=pos_lookup,
    neg_lookup=neg_lookup,
    base_negs=val_neg_imgs
)