import json
import numpy as np
import torch

from torch.utils.data import Dataset

class TokenIdsCache:
    '''Load the token ids matrix and token length array as memory-mapped NumPy arrays'''
    def __init__(self, prefix):
        meta = json.load(open(prefix + '_meta.json'))
        
        self.total_number = meta['total_number']
        self.max_seq_len = meta['embedding_size']
        self.type = meta['type']

        self.ids_arr = np.memmap(prefix + '.memmap', shape=(self.total_number, self.max_seq_len), dtype=np.dtype(self.type), mode="r")
        self.lengths_arr = np.load(prefix + '_length.npy')

        assert len(self.lengths_arr) == self.total_number
        
    def __len__(self):
        return self.total_number
    
    def __getitem__(self, item):
        return self.ids_arr[item, :self.lengths_arr[item]]


class SequenceDataset(Dataset):
    '''A PyTorch dataset for pre-processed token ids'''
    def __init__(self, ids_cache, max_seq_length):
        self.ids_cache = ids_cache
        self.max_seq_length = max_seq_length
        
    def __len__(self):  
        return len(self.ids_cache)

    def __getitem__(self, item):
        input_ids = self.ids_cache[item].tolist()
        seq_length = min(self.max_seq_length-1, len(input_ids)-1)
        # Estrai i token senza padding, mantenendo però l'ultimo, di chiususa
        input_ids = [input_ids[0]] + input_ids[1:seq_length] + [input_ids[-1]]
        attention_mask = [1] * len(input_ids)

        ret_val = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "id": item,
        }
        return ret_val

def pack_tensor_in_batch(list_of_lists, default, dtype, length=None):
    '''Convert a list of list into a 2D tensor batch'''
    batch_size = len(list_of_lists)
    length = length if length is not None else max(len(l) for l in list_of_lists)
    tensor = default * torch.ones((batch_size, length), dtype=dtype)
    for i, l in enumerate(list_of_lists):
        tensor[i, 0:len(l)] = torch.tensor(l, dtype=dtype)
    return tensor


def single_get_collate_function(max_seq_length, padding=False):
    cnt = 0
    def collate_function(batch):
        nonlocal cnt
        length = None
        if cnt < 10 or padding:
            length = max_seq_length
            cnt += 1

        input_ids      = [x["input_ids"]      for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]
        ids            = [x['id']             for x in batch]
        
        data = {
            "input_ids":      pack_tensor_in_batch(input_ids, default=1,      dtype=torch.int64, length=length),
            "attention_mask": pack_tensor_in_batch(attention_mask, default=0, dtype=torch.int64, length=length),
        }
        
        return data, ids
    return collate_function  
