from curses import erasechar
from torch.utils.data import Dataset, DataLoader
from functools import partial
import torch


class FinetuneDataset(Dataset):
  def __init__(self,
               data,
               tokenizer,
               col_article,
               col_summary
               ):
    self.tokenizer = tokenizer
    self.data = data
    self.col_article = col_article
    self.col_summary = col_summary
    self.sos = tokenizer.token_to_id("<s>")
    self.eos = tokenizer.token_to_id("</s>")

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    x = self.tokenizer.encode(
        self.data[self.col_article].values[idx]
        ).ids
    x = torch.tensor(x)
    target = self.tokenizer.encode(
        self.data[self.col_summary].values[idx]
        ).ids  
    target = torch.tensor(target)

    return {'input':x, 'target':target}


class FineTuneLoader(DataLoader):
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
    col_article, 
    col_summary, 
    batch_size = 8, 
    shuffle = False, 
    max_length = 750
    ):

    dataset = FinetuneDataset(data, tokenizer, col_article, col_summary)
    loader = cls(dataset, batch_size, shuffle, partial(collate_fn, max_length = max_length))
    return loader

def collate_fn(records, max_length = 750):
  max_len_input = min(max_length, max(len(record['input']) for record in records))
  max_len_target = max(len(record['target']) for record in records)

  attention_mask = torch.zeros(len(records), max_len_input)
  target = torch.zeros(len(records), max_len_target)
  input = torch.zeros(len(records), max_len_input)

  for i, ten in enumerate(records):
    len_ = min(max_len_input, len(ten['input']))
    input[i, :len_] += ten['input'][:len_]
    input[i][input[i]==0] = 1

    target[i, :len(ten['target'])] += ten['target']
    target[i][target[i]==0] = 1
    
    attention_mask[i, :len_] = 1

  lm_labels = target[:, 1:].clone()
  lm_labels[target[:, 1:] == 1] = -100

  target = target[:, :-1].contiguous()

  return {'input_ids':input.type(torch.LongTensor),
          'attention_mask':attention_mask.type(torch.LongTensor),
          'decoder_input_ids':target.type(torch.LongTensor),
          'labels':lm_labels.type(torch.LongTensor)}
