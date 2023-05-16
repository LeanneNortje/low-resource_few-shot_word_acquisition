#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import torchaudio
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from align import align_semiglobal, score_semiglobal

BACKEND = "nccl"
INIT_METHOD = "tcp://localhost:54321"

audio_segments_dir = Path('../../../QbERT/segments_flickr_yoruba')
# train_save_fn = '../data/train_for_preprocessing.npz'
ss_save_fn = '../support_set/support_set.npz'
support_set = np.load(ss_save_fn, allow_pickle=True)['support_set'].item()
files = "../../../Datasets/yfacc_v6"
train_fn = Path('../../../Datasets/yfacc_v6/Flickr8k_text/Flickr8k.token.train_yoruba.txt')
val_fn = Path('../../../Datasets/yfacc_v6/Flickr8k_text/Flickr8k.token.dev_yoruba.txt')
pam = np.load("pam.npy")

vocab = []
with open('../data/test_keywords.txt', 'r') as f:
    for keyword in f:
        vocab.append(' '.join(keyword.split()))

key = {}
id_to_word_key = {}
for i, l in enumerate(vocab):
    key[l] = i
    id_to_word_key[i] = l
    print(f'{i}: {l}')

np.savez_compressed(
    Path('../data/label_key'),
    key=key,
    id_to_word_key=id_to_word_key
)

s_audio = {}
s_wavs = []

for name in tqdm(support_set):

    _, img, wav, _, _, word, start, end, _ = support_set[name]
    # print(wav, start, end)

    fn = Path(wav).relative_to(*Path(wav).parts[:3]).with_suffix('.npz')
    fn = audio_segments_dir / fn

    query = np.load(fn)
    x = query["codes"][query["boundaries"][:-1]]
    # print(x)

    x0, = np.where(query["boundaries"] <= int(float(start)*50))
    x0 = x0[-1]
    xn, = np.where(query["boundaries"] >= int(float(end)*50))
    xn = xn[0]
    x = x[x0-1:xn+1]

    if word not in s_audio: s_audio[word] = []
    s_audio[word].append(x)
    s_wavs.append(Path(wav).stem)

# with open(Path('../data/train.json'), 'r') as f:
#     train = json.load(f)

# with open(val_fn, 'r') as f:
#     val = json.load(f)

image_base = Path('../../../Datasets/Flicker8k_Dataset')
english_base = Path('../../../Datasets/flickr_audio/wavs')
yoruba_base = Path('../../../Datasets/yfacc_v6/flickr_audio_yoruba_train')

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

# for w in train:
#     if w not in counts: counts[w] = 0
#     for im, eng, yor in train[w]:
#         data.append((im, eng, yor, w))
#         print(im, eng, yor, w)
#         counts[w] += 1
#         break
#     break
        
# for w in val:
#     for im, eng, yor in val[w]:
#         data.append((im, eng, yor, w))
#         counts[w] += 1

# for w in counts:
#     print(w, counts[w])

query_scores = {}
record = {}

for q_word in s_audio:

    id = key[q_word]
    for entry in tqdm(data, desc=f'{q_word}({id})'):

        wav = entry[2]
        wav_name = Path(wav).stem
        if wav_name in s_wavs: continue

        if id not in query_scores: query_scores[id] = {'values': [], 'wavs': []}
        this_fn = Path(wav).relative_to(*Path(wav).parts[:5]).with_suffix('.npz')
        this_fn = audio_segments_dir / this_fn
        for fn in this_fn.parent.rglob(f'*{this_fn.stem}.npz'):
            test = np.load(fn)
            y = test["codes"][test["boundaries"][:-1]]

            max_score = -np.inf
            for x in s_audio[q_word]:
                path, p, q, score = align_semiglobal(x, y, pam, 3)
                indexes, = np.where(np.array(p) != -1)
                if len(indexes) != 0:
                    start, end = indexes[1], indexes[-1]
                    norm_score = score / (end - start)

                    if norm_score > max_score: 
                        max_score = norm_score
                        if str(wav) not in record: record[str(wav)] = {}
                        record[str(wav)][id] = (path, start, end, p, q, indexes)
            query_scores[id]['values'].append(max_score)
            query_scores[id]['wavs'].append(str(wav))   
    # break
        
            # if len(query_scores[id]['values']) == 1900: break

# for id in query_scores:
#     print(id, len(query_scores[id]['values']), len(query_scores[id]['wavs']))

save_dir = Path('segment_examples')
audio_dir = Path("../../Datasets/yfacc_v6")
top_N = 100
newly_labeled = {}
frame_info = {}

for id in query_scores:
    indices = np.argsort(query_scores[id]['values'])[::-1]
    if id not in newly_labeled: newly_labeled[id] = []
    if id not in frame_info: frame_info[id] = []
    
    for i in range(len(indices)):
        if len(newly_labeled[id]) == top_N: break
        wav = Path(query_scores[id]['wavs'][indices[i]])
        wav_name = wav.stem

        fn = save_dir / Path(id_to_word_key[id]) / wav_name
        fn.parent.mkdir(parents=True, exist_ok=True)
        this_fn = Path(wav).relative_to(*Path(wav).parts[:5]).with_suffix('.npz')
        for segment_fn in audio_segments_dir.rglob(f'*{this_fn.stem}.npz'):
            
            test = np.load(segment_fn)
            path, start, end, p, q, indexes = record[str(wav)][id]
            _, b0 = path[start - 1]
            _, bT = path[end]
            w0, wT = 0.02 * test["boundaries"][b0 - 1], 0.02 * test["boundaries"][bT]
            offset = int(w0 * 48000)
            frames = int(np.abs(wT - w0) * 48000)
            if frames > 0:
                aud, sr = torchaudio.load(wav, frame_offset=offset, num_frames=frames)

                # print(frames, aud.size())
                if frames == aud.size(1):
                    
                    torchaudio.save(fn.with_suffix('.wav'), aud, sr)

                    newly_labeled[id].append(wav)
                    
                    frame_info[id].append((wav, offset, wT*16000))
                    # print(len(newly_labeled[id]))
                    # print(i)

for id in newly_labeled:
    print(id, len(newly_labeled[id]))

# np.savez_compressed(
#     Path("../data/sampled_audio_data"), 
#     data=newly_labeled
# )

np.savez_compressed(
    Path("../data/sampled_audio_frame_info"), 
    data=frame_info
)