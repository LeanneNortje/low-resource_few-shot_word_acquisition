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

K = 100
num_episodes = 1000

def load_fn(fn, word_points, vocab):
    count = {}
    with open(fn, 'r') as f:
        data = json.load(f)

    data = data['data']

    for entry in tqdm(data):
        image = entry['image']

        for caption in entry['captions']:
            
            for word in vocab:
                if re.search(word, caption['text'].lower()) is not None:
                    if word not in word_points:
                        word_points[word] = []
                    word_points[word].append((image, caption['wav'], caption['speaker']))
                    if word not in count: count[word] = 0
                    count[word] += 1

    print('Word counts')
    for word in count:
        print(f'{word}: {count[word]}')
                    

files = "../../Datasets/spokencoco/SpokenCOCO"
train_fn = Path(files) / 'SpokenCOCO_train.json'
train = {}

vocab = []
with open('../data/test_keywords.txt', 'r') as f:
    for keyword in f:
        vocab.append(' '.join(keyword.split()))

annotations_fn = Path("../../Datasets/spokencoco/captions_train2017.json")
f = json.load(open(annotations_fn))
annotations = f['annotations']
im_annotations = f['images']
print(len(annotations), len(im_annotations))

save_dir = Path('../support_set')
support_set = np.load(Path(save_dir / Path('support_set_100.npz')), allow_pickle=True)['support_set'].item()

for entry in annotations: 
    print(entry)
    break

for entry in im_annotations: 
    print(entry)
    break

train_list = set()
for entry in im_annotations: 
    train_list.add(entry['file_name'])

annotations_fn = Path("../../Datasets/spokencoco/captions_val2017.json")
f = json.load(open(annotations_fn))
annotations = f['annotations']
im_annotations = f['images']
print(len(annotations), len(im_annotations))

for entry in annotations: 
    print(entry)
    break

for entry in im_annotations: 
    print(entry)
    break

val_list = set()
for entry in im_annotations: 
    val_list.add(entry['file_name'])

support_set_list = set()
for name in support_set:
    wav, im, spkr, start, stop, word = support_set[name]
    support_set_list.add(im.split('_')[-1])
    
print(len(train_list), len(support_set_list), len(support_set_list - train_list), len(train_list - support_set_list))
print(len(val_list), len(support_set_list), len(support_set_list - val_list))
