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

def sub_sample(sup, K):
    new_set = {}
    reduced = {}

    for w in sup:
        instances = np.random.choice(np.arange(0, len(sup[w])), K, replace=False)
        reduced[w] = [sup[w][i] for i in instances]
    
    for w in reduced:
        for wav, img, spkr, start, stop, word in reduced[w]:
            if w != word: print("Bug")
            wav_name = Path(wav).stem
            new_set[wav_name] = (wav, img, spkr, start, stop, word)

    return reduced, new_set

support_set = np.load('../support_set/support_set_100.npz', allow_pickle=True)['support_set'].item()

save_dir = Path('../support_set')

support_set_50 = {}
support_set_10 = {}
support_set_5 = {}

per_word = {}
for wav_name in support_set:
    wav, img, spkr, start, stop, word = support_set[wav_name]
    if word not in per_word: per_word[word] = []
    per_word[word].append(support_set[wav_name])

for w in per_word:
    print(w, len(per_word[w]))

per_word, support_set_50 = sub_sample(per_word, 50)
for w in per_word:
    print(w, len(per_word[w]))

np.savez_compressed(
    save_dir / Path('support_set_50'), 
    support_set=support_set_50
    )

per_word, support_set_10 = sub_sample(per_word, 10)
for w in per_word:
    print(w, len(per_word[w]))

np.savez_compressed(
    save_dir / Path('support_set_10'), 
    support_set=support_set_10
    )

per_word, support_set_5 = sub_sample(per_word, 5)
for w in per_word:
    print(w, len(per_word[w]))

np.savez_compressed(
    save_dir / Path('support_set_5'), 
    support_set=support_set_5
    )