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
import textgrids

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize, sent_tokenize

classes = set()

for txt_grid in Path('../../Datasets/yfacc_v6/Flickr8k_alignment').rglob('*.TextGrid'):
    if str(txt_grid) == '../../Datasets/yfacc_v6/Flickr8k_alignment/3187395715_f2940c2b72_0.TextGrid': continue
    grid = textgrids.TextGrid(txt_grid)
    wav = 'S001_' + txt_grid.stem
    im = '_'.join(txt_grid.stem.split('_')[0:2])

    for interval in grid['words']:
        
        x = str(interval).split()
        label = str(interval).split('"')[1]
        start = x[-2].split('=')[-1]
        dur = x[-1].split('=')[-1].split('>')[0]

        classes.add(label)

print(classes)

concepts = []
with open('./data/test_keywords.txt', 'r') as f:
    for keyword in f:
        concepts.append(keyword.strip())

yoruba_vocab = set()
with open(Path('../../Datasets/yfacc_v6/Flickr8k_text/eng_yoruba_keywords.txt'), 'r') as f:
    for line in f:
        e, y = line.strip().split(', ')
        if e in concepts:
            yoruba_vocab.add(y)
print(yoruba_vocab)
print(classes - yoruba_vocab)
