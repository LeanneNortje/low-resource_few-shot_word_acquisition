#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2023
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

from pathlib import Path
import numpy as np
import json
import re 
import textgrids

key = np.load(Path('../data/label_key.npz'), allow_pickle=True)['id_to_word_key'].item()
for word in key:
    print(f'{key[word]:<3}: {word}')

vocab = []
with open('../data/test_keywords.txt', 'r') as f:
    for keyword in f:
        vocab.append(' '.join(keyword.split()))

transcriptions = {}
prev = ''
prev_wav = ''
prev_start = 0

translation = {}
yoruba_vocab = []
with open(Path('../../../Datasets/yfacc_v6/Flickr8k_text/eng_yoruba_keywords.txt'), 'r') as f:
    for line in f:
        e, y = line.strip().split(', ')
        if e in vocab:
            translation[e] = y
            yoruba_vocab.append(y)
for e in translation:
    print(e, translation[e])

for split in ['train', 'dev']:
    fn = Path(f'../../../Datasets/yfacc_v6/Flickr8k_text/Flickr8k.token.{split}_yoruba.txt')
    with open(fn, 'r') as f:
        for line in f:
            parts = line.split()
            wav = 'S001_' + parts[0].split('.')[0] + '_' + parts[0].split('#')[-1]
            text = ' ' + ' '.join(parts[1:]) + ' '
            for label in yoruba_vocab:
                if wav not in transcriptions: transcriptions[wav] = []
                if re.search(' ' + label + ' ', text.lower()) is not None:
                    transcriptions[wav].append(label)

ss_save_fn = '../support_set/support_set.npz'
support_set = np.load(ss_save_fn, allow_pickle=True)['support_set'].item()
s_aud = []
for name in support_set:
    s_aud.append(name)
    
train_id_lookup = np.load(Path("../data/train_lookup.npz"), allow_pickle=True)['lookup'].item()
val_id_lookup = np.load(Path("../data/val_lookup.npz"), allow_pickle=True)['lookup'].item()
train_neg_id_lookup = np.load(Path("../data/train_lookup.npz"), allow_pickle=True)['neg_lookup'].item()
train_audio_per_keyword_acc = {}
results = {}

for id in train_id_lookup:

    pred_word = translation[key[id]]

    for name in train_id_lookup[id]['audio']: 
        if name in support_set:  continue
        if name not in results: results[name] = {'pred': [], 'gt': set()}
        results[name]['pred'].append(pred_word)
        results[name]['gt'].update(transcriptions[name])

for id in val_id_lookup:

    pred_word = translation[key[id]]

    for name in val_id_lookup[id]['audio']: 
        # name = 'S001_' + name
        if name not in results: results[name] = {'pred': [], 'gt': set()}
        results[name]['pred'].append(pred_word)
        results[name]['gt'].update(transcriptions[name])

for i, name in enumerate(results):
    
    for p_w in results[name]['pred']: 
        if p_w not in train_audio_per_keyword_acc: train_audio_per_keyword_acc[p_w] = {"tp": 0, "fn": 0, "fp": 0, 'counts': 0}
        
        if p_w in results[name]['gt']: train_audio_per_keyword_acc[p_w]['tp'] += 1
        else: train_audio_per_keyword_acc[p_w]['fp'] += 1
        train_audio_per_keyword_acc[p_w]['counts'] += 1

    for g_w in results[name]['gt']: 
        if g_w not in results[name]['pred']: train_audio_per_keyword_acc[p_w]['fn'] += 1

t_tp = 0
t_fp = 0
t_fn = 0
print(f'Training accuracies:')
for word in train_audio_per_keyword_acc:
    tp = train_audio_per_keyword_acc[word]['tp']
    t_tp += tp
    fp = train_audio_per_keyword_acc[word]['fp']
    t_fp += fp
    fn = train_audio_per_keyword_acc[word]['fn']
    t_fn += fn
    pres = tp / (tp + fp)
    recall = tp / (tp + fn)
    c = train_audio_per_keyword_acc[word]['counts']
    print(f'{word:<10}\t Precision: {100*pres:.2f}% ({c})')
pres = t_tp / (t_tp + t_fp)
recall = t_tp / (t_tp + t_fn)
a = 'Overall'
print(f'{a:<10}\t Precision: {100*pres:.2f}%')