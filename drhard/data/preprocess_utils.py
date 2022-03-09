import os
import json
import numpy as np
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, PreTrainedTokenizer

# check for correct transformers version
if int(transformers.__version__.split(".")[0])>=3:
    raise RuntimeError("Please use Transformers Libarary version 2 for preprossing because we find RobertaTokenzier behaves differently between version 2 and 3/4")

def pad_input_ids(input_ids: list[int], max_length: int, pad_on_left: bool=False, pad_token: int=0) -> list[int]:
    '''Cut down down to max_length or pad input_ids up to max_length with a given pad_token, on the left or on the right (default: right)'''
    padding_length = max_length - len(input_ids)

    if padding_length <= 0:
        input_ids = input_ids[:max_length]
    else:
        if pad_on_left:
            input_ids = [pad_token] * padding_length + input_ids
        else:
            input_ids = input_ids + [pad_token] * padding_length
    return input_ids

def _compute_token_ids(text: str, tokenizer: PreTrainedTokenizer, max_length: int):
    '''Process a string of text, returning the the padded token_ids (obtained with the tokenizer) up to max_length and the original length (up to max_length)'''

    token_ids = tokenizer.encode(text, add_special_tokens=True, max_length=max_length, truncation=True)
    token_ids_len = min(len(token_ids), max_length)
    padded_token_ids = pad_input_ids(token_ids, max_length)

    return padded_token_ids, token_ids_len


def query_preprocessing_fn(line: str, tokenizer: PreTrainedTokenizer, max_length: int):
    '''Process a MSMARCO query string, returing the qid as int, the padded token_ids (obtained with the tokenizer) up to max_length and the original length (up to max_length)'''
    qid, query = line.split('\t')
    qid = int(qid)  
    padded_token_ids, token_ids_len = _compute_token_ids(query, tokenizer, max_length)
    return qid, padded_token_ids, token_ids_len


def msmarco_document_preprocessing_fn(line: str, tokenizer: PreTrainedTokenizer, max_length: int):
    '''Process a MSMARCO document string, returing the did as int, the padded token_ids (obtained with the tokenizer) up to max_length and the original length (up to max_length)'''
    did, url, title, body = line.split('\t')
    did = int(did[1:])  # remove "D"
    # NOTE: This linke is copied from ANCE, but I think it's better to use <s> as the separator, 
    # TODO: could use tokenizer.sep_token, but it needs re-training
    text = url + "<sep>" + title + "<sep>" + body
    # keep only first 10000 characters, should be sufficient for any experiment that uses less than 500 - 1k tokens
    text = text[:10000]

    padded_token_ids, token_ids_len = _compute_token_ids(text, tokenizer, max_length)
    return did, padded_token_ids, token_ids_len


def msmarco_passage_preprocessing_fn(line: str, tokenizer: PreTrainedTokenizer, max_length: int):
    '''Process a MSMARCO passage string, returing the pid as int, the padded token_ids (obtained with the tokenizer) up to max_length and the original length (up to max_length)'''
    pid, text = line.split('\t')
    pid = int(pid)
    # keep only first 10000 characters, should be sufficient for any experiment that uses less than 500 - 1k tokens
    text = text[:10000]
    
    padded_token_ids, token_ids_len = _compute_token_ids(text, tokenizer, max_length)
    return pid, padded_token_ids, token_ids_len


def tokenize_to_file(model_name_or_path: str, in_path: str, output_dir: str, 
                     preprocessing_fn, begin_idx: int, end_idx: int, 
                     max_length: int, pid: int=0) -> None:
    '''
    Process a given input file (MSMarco queries, documents or passages) and compute the token ids of the whole content with a given model tokenizer.
    The output consists of 3 file:
    * ids.memmap: a NumPy memory-mapped file containing an array of qid/did/pid as ints
    * token_ids.memmap: a NumPy memory-mapped file containing an array of token ids arrays, as ints
    * lengths.memmap_ a NumPy memory-mapped file containing the original un-padded lengths of the documents as ints
    '''

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, do_lower_case = True)
    os.makedirs(output_dir, exist_ok=True)
    
    data_cnt = end_idx - begin_idx

    ids_array          = np.memmap(os.path.join(output_dir, "ids.memmap"),       shape=(data_cnt, ),           mode='w+', dtype=np.int32)
    token_ids_array    = np.memmap(os.path.join(output_dir, "token_ids.memmap"), shape=(data_cnt, max_length), mode='w+', dtype=np.int32)
    token_length_array = np.memmap(os.path.join(output_dir, "lengths.memmap"),   shape=(data_cnt, ),           mode='w+', dtype=np.int32)

    pbar = tqdm(total=data_cnt, desc="Tokenizing", position=pid, leave=False)
    for idx, line in enumerate(open(in_path, 'r')):
        if idx < begin_idx:
            continue
        if idx >= end_idx:
            break
        qid_or_pid, token_ids, length = preprocessing_fn(line, tokenizer, max_length)
        write_idx = idx - begin_idx
        ids_array[write_idx] = qid_or_pid
        token_ids_array[write_idx, :] = token_ids
        token_length_array[write_idx] = length
        pbar.update(1)
    pbar.close()
    assert write_idx == data_cnt - 1


def dump_json(output, total, dim):
    meta = {
        'type': 'int32', 
        'total_number': total,
        'embedding_size': dim}
    with open(output, 'w') as f:
        json.dump(meta, f)
