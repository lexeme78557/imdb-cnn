CNN_PAD = '<PAD>'
CNN_END = '<END>'
CNN_UNK = '<UNK>'

import torch
from torch.utils import data
from collections import defaultdict

class TextDataset(data.Dataset):
    def __init__(self, examples, split, threshold, max_len, idx2word=None, word2idx=None):
        self.examples = examples
        assert split in {'train', 'val', 'test'}
        self.split = split
        self.threshold = threshold
        self.max_len = max_len

        # Dictionaries
        self.word2idx = word2idx # Mapping of word to index
        self.idx2word = idx2word # Mapping of index to word
        if split == 'train':
            self.build_dictionary()
        self.vocab_size = len(self.word2idx)
        
        # Convert text to indices
        self.textual_ids = []
        self.convert_text()

    
    def build_dictionary(self): 
        '''
        Build the dictionaries idx2word and word2idx. This is only called when split='train', as these
        dictionaries are passed in to the __init__(...) function otherwise. Be sure to use self.threshold
        to control which words are assigned indices in the dictionaries.
        Returns nothing.
        '''
        assert self.split == 'train'
        
        # Don't change this
        self.idx2word = {0:CNN_PAD, 1:CNN_END, 2: CNN_UNK}
        self.word2idx = {CNN_PAD:0, CNN_END:1, CNN_UNK: 2}

        counter = dict()
        tmp = dict()
        words = []
        for i in range(len(self.examples)):
            for w in (self.examples[i][1]):
              words.append(w)

        for (i, word) in enumerate(words):
            word = word.lower()
            if (word not in counter):
                counter[word] = 1
            else:
                counter[word] += 1
            tmp[i] = word

        allowed = dict()
        for (index, value) in tmp.items():
            if (counter[value] >= self.threshold):
                allowed[value] = index
        count = 3
        for (k, v) in allowed.items():
            self.idx2word[count] = k
            count += 1
        for (k, v) in self.idx2word.items():
            self.word2idx[v] = k


    def convert_text(self):
        '''
        Convert each review in the dataset (self.examples) to a list of indices, given by self.word2idx.
        Store this in self.textual_ids; returns nothing.
        '''
        for reviews in self.examples:
            tmp = []
            label = 1 if reviews[0]=='pos' else 0
            for w in reviews[1]:
                if w not in self.word2idx:
                   tmp.append(2)
                else:
                   tmp.append(self.word2idx[w])
            tmp.append(1)
            self.textual_ids.append((label, tmp))

    def get_text(self, idx):
        '''
        Return the review at idx as a long tensor (torch.LongTensor) of integers corresponding to the words in the review.
        You may need to pad as necessary (see above).
        '''
        rev = self.textual_ids[idx][1]
        if (len(rev) >= self.max_len):
            rev = rev[:self.max_len]
        else:
            rev = rev + [0] * (self.max_len - len(rev))
        
        return torch.as_tensor(rev).long()
    
    def get_label(self, idx):
        '''
        This function should return the value 1 if the label for idx in the dataset is 'positive', 
        and 0 if it is 'negative'. The return type should be torch.LongTensor.
        '''
        label = self.textual_ids[idx][0]
        return torch.as_tensor(label).long()

    def __len__(self):
        '''
        Return the number of reviews (int value) in the dataset
        '''
        return (len(self.examples))
    
    def __getitem__(self, idx):
        '''
        Return the review, and label of the review specified by idx.
        '''
        return self.get_text(idx), self.get_label(idx)