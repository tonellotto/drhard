import os
import json
import pickle
import shutil
import argparse
import subprocess
import multiprocessing
import numpy as np

from . import preprocess_utils as pp

# conda activate drhard-tok
# python -m drhard.data.preprocess_collection --data_type 1 --threads 8 --input_path=/data2/DRhard/data/passage/dataset --output_path=./merda

def multi_file_process(args, num_process, in_path, out_path, preprocessing_fn):

    num_docs = int(subprocess.check_output(["wc", "-l", in_path]).decode("utf-8").split()[0])
    num_docs = 80_000 # for debugging
    print(f"Number of documents to process {num_docs}")

    run_arguments = []
    splits_dir = []
    for i in range(num_process):
        begin_idx = round(num_docs * i / num_process)
        end_idx   = round(num_docs * (i+1) / num_process)
        output_dir = f"{out_path}_split_{i}"
        run_arguments.append((args.model_name_or_path, in_path, output_dir, preprocessing_fn, begin_idx, end_idx, args.max_seq_length, i))
        splits_dir.append(output_dir)

    pool = multiprocessing.Pool(processes=num_process)
    pool.starmap(pp.tokenize_to_file, run_arguments)
    pool.close()
    pool.join()

    return splits_dir, num_docs


def preprocess(args):

    print('start processing splits')
    if args.data_type == 0:
        in_passage_path = os.path.join(args.input_path, "msmarco-docs.tsv")
        preprocessing_fn = pp.msmarco_document_preprocessing_fn
    else:
        in_passage_path = os.path.join(args.input_path, "collection.tsv")
        preprocessing_fn = pp.msmarco_passage_preprocessing_fn

    out_passage_path = args.output_path

    splits_dir_lst, num_docs = multi_file_process(args, args.threads, 
                                                  in_passage_path, out_passage_path, 
                                                  preprocessing_fn)

    print('start merging splits')
    token_ids_array = np.memmap(os.path.join(out_passage_path, "collection.memmap"), shape=(num_docs, args.max_seq_length), mode='w+', dtype=np.int32)
    pid2offset = {}
    token_length_array = []

    idx = 0
    for split_dir in splits_dir_lst:
        split_pids_array         = np.memmap(os.path.join(split_dir, "ids.memmap"),       mode='r', dtype=np.int32)
        split_token_ids_array    = np.memmap(os.path.join(split_dir, "token_ids.memmap"), mode='r', dtype=np.int32).reshape(len(split_pids_array), -1)
        split_token_length_array = np.memmap(os.path.join(split_dir, "lengths.memmap"),   mode='r', dtype=np.int32)

        for pid, token_ids, length in zip(split_pids_array, split_token_ids_array, split_token_length_array):
            token_ids_array[idx, :] = token_ids
            token_length_array.append(length) 
            pid2offset[pid] = idx
            idx += 1
    assert len(token_length_array) == len(token_ids_array) == idx

    np.save(os.path.join(out_passage_path, "collection_length.npy"), np.array(token_length_array))
    pp.dump_json(os.path.join(out_passage_path, "collection_meta.json"), idx, args.max_seq_length)
    print(f"Total lines written: {idx}")
    
    with open( os.path.join(out_passage_path, "collection_pid2offset.pickle"), 'wb') as handle:
        pickle.dump(pid2offset, handle, protocol=4)
    
    for split_dir in splits_dir_lst:
        shutil.rmtree(split_dir)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path",                                 type=str)
    parser.add_argument("--output_path",                                type=str)

    parser.add_argument("--model_name_or_path", default="roberta-base", type=str)
    parser.add_argument("--data_type",          default=1,              type=int, help="0 for doc, 1 for passage")

    parser.add_argument("--max_seq_length",     default=512,            type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_query_length",   default=64,             type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_doc_character",  default=10000,          type=int, help="used before tokenizer to save tokenizer latency")

    parser.add_argument("--threads", default=multiprocessing.cpu_count(), type=int)

    args = parser.parse_args()

    if os.path.exists(os.path.join(args.output_path, "collection.memmap")):
        print(f"preprocessed data already exist in {args.output_path}, exit preprocessing")
    else:
        os.makedirs(args.output_path, exist_ok=True)
        preprocess(args)