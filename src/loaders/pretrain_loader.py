from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import numpy as np
import random
import torch
import re


class PretrainDataset(Dataset):
  def __init__(self,
               data,
               tokenizer,
               text_col = 'text',
               max_length = 300,
               sentence_permutation = True,
               token_masking = False,
               token_deletion = False, 
               text_infilling = True):
    
    self.data = data
    self.tokenizer = tokenizer
    self.text_col = text_col
    self.max_length = max_length
    self.sentence_permutation = sentence_permutation
    self.token_deletion = token_deletion
    self.token_masking = token_masking
    self.text_infilling = text_infilling
    self.get_special_tokens_ids()

  def get_special_tokens_ids(self):
    self.mask = self.tokenizer.token_to_id("<mask>")
    self.eos = self.tokenizer.token_to_id("</s>")
    self.sos = self.tokenizer.token_to_id("<s>")
    self.pad = self.tokenizer.token_to_id("<pad>")

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):

    x = self.data[self.text_col].values[idx]
    target = self.tokenizer.encode(x).ids

    if self.sentence_permutation:
        split_regex = re.compile(r'[.|!|?|…]')
        sentenced = list(filter(lambda t: t, [t.strip() for t in split_regex.split(x)]))
        x = random.sample(sentenced, len(sentenced))
        x = ' '.join(x)
    x = self.tokenizer.encode(x).ids

    if self.token_deletion:
        len_ = len(x)
        idx = random.sample(np.arange(len_).tolist(), int(len_*0.15))
        for i in idx:
          x.remove(x[i])

    if self.token_masking:
        x = np.array(x)
        len_ = len(x)
        idx = random.sample(np.arange(len_).tolist(), int(len_*0.15))
        x[idx] = self.mask
        x = x.tolist()

    if self.text_infilling: 
        x = np.array(x)
        number_of_spans = random.choice(range(5,8))
        idx = random.sample(np.arange(len(x)).tolist(), number_of_spans)
        poissons = np.random.poisson(3, number_of_spans)
        new_idx = zip(idx, poissons)
        copy_idx = deepcopy(new_idx)
        x[np.concatenate([np.arange(i, min(i+j+1, len(x))) for i, j in new_idx])] = -1
        x = x.tolist()

        for i, _ in list(copy_idx):
          x.insert(i, self.mask)
        x = np.array(x)
        x = x[x!=-1].tolist()
    
    attention_mask = [1]*len(x)
    
    if len(x)>= self.max_length:
      x = x[:self.max_length-1]
      x = x + [self.eos]
      attention_mask = attention_mask[:self.max_length]
    else:
      x += [self.pad]*(self.max_length-len(x))
      attention_mask += [0]*(self.max_length-len(attention_mask))
    
    if len(target)>= self.max_length:
      target = target[:self.max_length-1]
      target = target + [self.eos]
    else:
      target += [self.pad]*(self.max_length-len(target))
    
    x = torch.LongTensor(x)
    attention_mask = torch.LongTensor(attention_mask)
    target = torch.LongTensor(target)
    lm_labels = target[1:].clone()
    lm_labels[target[1:] == 1] = -100

    return {
            'input_ids': x,
            'attention_mask': attention_mask,
            'decoder_input_ids': target,
            'lm_labels': lm_labels
            }


class PretrainLoader(DataLoader):
  def __init__(
    self, 
    dataset, 
    batch_size = 8, 
    shuffle = False, 
    collate_fn = None
    ):
      super().__init__(
        dataset, 
        batch_size = batch_size, 
        shuffle = shuffle, 
        collate_fn = collate_fn
        )

  @classmethod
  def load(
    cls, 
    data, 
    tokenizer, 
    max_length = 750, 
    batch_size = 8,
    shuffle = False,
    sentence_permutation = True,
    token_masking = False,
    token_deletion = False,
    text_infilling = True
    ):

    dataset = PretrainDataset(
      data = data, 
      tokenizer = tokenizer, 
      max_length = max_length,
      sentence_permutation = sentence_permutation,
      token_masking = token_masking,
      token_deletion = token_deletion,
      text_infilling=text_infilling
      )
    loader = cls(dataset, batch_size, shuffle)
    return loader
