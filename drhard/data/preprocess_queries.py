import os
import pickle
import shutil
import argparse
import subprocess
import multiprocessing
import numpy as np

from tqdm import tqdm
from pathlib import Path
from . import preprocess_utils as pp


# conda activate drhard-tok
# questo codice è quasi identico a preprocess_collection, cambiano solo i pathname e la lunghezza massima (e nessuna distinzione tra documenti e passaggi)

def multi_file_process(args, num_process: int, in_path: str, out_path: str, preprocessing_fn):

    num_docs = int(subprocess.check_output(["wc", "-l", in_path]).decode("utf-8").split()[0])
    #num_docs = 80_000 # for debugging
    print(f"Number of documents to processline {num_docs}")

    run_arguments = []
    splits_dir = []
    for i in range(num_process):
        begin_idx = round(num_docs * i / num_process)
        end_idx   = round(num_docs * (i+1) / num_process)
        output_dir = f"{out_path}_split_{i}"
        run_arguments.append((args.model_name_or_path, in_path, output_dir, preprocessing_fn, begin_idx, end_idx, args.max_query_length, i))
        splits_dir.append(output_dir)

    pool = multiprocessing.Pool(processes=num_process, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
    pool.starmap(pp.tokenize_to_file, run_arguments)
    pool.close()
    pool.join()

    return splits_dir, num_docs


def _count_valid_queries(query_file: str):

    return int(subprocess.check_output(["wc", "-l", query_file]).decode("utf-8").split()[0])


def preprocess(args):

    valid_query_num = _count_valid_queries(args.query_file)

    print('start processing splits')
    preprocessing_fn = pp.query_preprocessing_fn
    splits_dir_lst, _ = multi_file_process(args, args.threads, 
                                           args.query_file, args.output_path, 
                                           preprocessing_fn)

    print('start merging splits')

    token_ids_array = np.memmap(args.output_prefix + ".memmap", shape=(valid_query_num, args.max_query_length), mode='w+', dtype=np.int32)
    qid2offset = {}
    token_length_array = []

    idx = 0
    for split_dir in splits_dir_lst:
        split_qids_array         = np.memmap(os.path.join(split_dir, "ids.memmap"),       mode='r', dtype=np.int32)
        split_token_ids_array    = np.memmap(os.path.join(split_dir, "token_ids.memmap"), mode='r', dtype=np.int32).reshape(len(split_qids_array), -1)
        split_token_length_array = np.memmap(os.path.join(split_dir, "lengths.memmap"),   mode='r', dtype=np.int32)
        
        for qid, token_ids, length in zip(split_qids_array, split_token_ids_array, split_token_length_array):
            token_ids_array[idx, :] = token_ids
            token_length_array.append(length) 
            qid2offset[qid] = idx
            idx += 1
    assert len(token_length_array) == len(token_ids_array) == idx

    np.save(args.output_prefix + "_length.npy", np.array(token_length_array))
    pp.dump_json(args.output_prefix + "_meta.json", idx, args.max_query_length)

    print("Total lines written: " + str(idx))
    
    qid2offset_path = args.output_prefix + '_qid2offset.pickle'
    with open(qid2offset_path, 'wb') as handle:
        pickle.dump(qid2offset, handle, protocol=4)

    for split_dir in splits_dir_lst:
        shutil.rmtree(split_dir)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--query_file",         required=True,          type=str)
    parser.add_argument("--output_path",        required=True,          type=str)

    parser.add_argument("--model_name_or_path", default="roberta-base", type=str)
    parser.add_argument("--max_query_length",   default=32,             type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--threads", default=multiprocessing.cpu_count(), type=int)

    args = parser.parse_args()

    args.output_prefix = os.path.join(args.output_path, Path(args.query_file).stem)

    if os.path.exists(args.output_prefix + ".memmap"):
        print(f"preprocessed data already exist in {args.output_path}, exit preprocessing")
    else:
        os.makedirs(args.output_path, exist_ok=True)
        preprocess(args)
