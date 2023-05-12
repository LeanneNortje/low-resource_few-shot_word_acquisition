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
            if label not in alignments[wav]: alignments[wav][label] = (int(float(start)*50), int(float(dur)*50))
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

    # sentence = ''
    # for interval in grid['words']:

    #     label = str(interval).split('"')[1]
    #     sentence += label
    #     sentence += ' '

    # for e_v in vocab:
        # v = translation[e_v]
        # print()
        # if re.search(v, sentence) is not None:

            # parts = v.split()
            # recontructed = ''
            # word_start = -1
    
    for interval in grid['words']:
        
        x = str(interval).split()
        label = str(interval).split('"')[1]
        start = x[-2].split('=')[-1]
        dur = x[-1].split('=')[-1].split('>')[0]
        # print(label, start, stop, label in vocab)
        
        # if label == parts[0]:
        #     word_start = float(start)
        #     recontructed += label
        # if start != -1 and label in parts:
        #     recontructed += ' '
        #     recontructed += label
        #     if v == recontructed:
        #         end = float(stop)
        if label in yoruba_vocab:
            print(label)
            if wav not in yoruba_alignments: yoruba_alignments[wav] = {}
            if label not in yoruba_alignments[wav]: yoruba_alignments[wav][label] = (int(float(start)*50), int(float(dur)*50))

support_set = {}

# ##################################
# # Support set 
# ##################################
fn = Path('../data/test.json')
with open(fn, 'r') as f:
    train = json.load(f)

for word in vocab:
    # filtered = [(i, w, s) for i, w, s in train[word] if Path(w).stem in alignments and Path(w).stem not in support_set]
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
    
    for im, wav, yor in [filtered[i] for i in instances]:
        name = Path(wav).stem
        if name in support_set or count == K or name not in yoruba_alignments: continue
        dur = int((float(alignments[name][word][1])*0.02)*16000)
        offset = int(float(alignments[name][word][0])*16000*0.02)
        aud, sr = torchaudio.load(Path('..') / wav, frame_offset=offset, num_frames=dur)
        torchaudio.save(Path('..') / Path('temp.wav'), aud, sr)
        ans = input(f'{count} / {K} {word}: ')
        if ans == 'y':
            support_set[name] = (wav, im, alignments[name][word][0], alignments[name][word][1], word)
            save_name = Path(wav).stem  + '_' + word + '.wav'
            out_path = save_dir / Path(wav).parent / Path(save_name)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            torchaudio.save(out_path, aud, sr)

            # y_word = translation[word]
            # y_dur = int((float(yoruba_alignments[name][y_word][1])*0.02 )*48000)
            # y_offset = int(float(yoruba_alignments[name][y_word][0])*48000*0.02)
            # aud, sr = torchaudio.load(Path('..') / yor, frame_offset=y_offset, num_frames=y_dur)

            # save_name = Path(yor).stem  + '_' + y_word + '.wav'
            # out_path = save_dir / Path(wav).parent / Path(save_name)
            # out_path.parent.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(
                save_dir / Path('support_set_eng'), 
                support_set=support_set
                )
            count += 1
        if count == K: break

np.savez_compressed(
    save_dir / Path('support_set_eng'), 
    support_set=support_set
    )