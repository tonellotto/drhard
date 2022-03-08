import os
import pickle
import shutil
import argparse
import subprocess
import multiprocessing
import numpy as np

from . import preprocess_utils as pp


# conda activate drhard-tok
# python preprocess_collection.py --data_type 1 --threads 8 --input_path=./data/passage/dataset --output_path=./merda
# python preprocess_queries.py  --threads 8 --query_file=./data/passage/dataset/queries.2019.manual-newID.tsv --pid2offset_file=./merda_pid2offset.pickle --output_path=./cazzo

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

    pool = multiprocessing.Pool(processes=num_process)
    pool.starmap(pp.tokenize_to_file, run_arguments)
    pool.close()
    pool.join()

    return splits_dir, num_docs


def count_valid_queries(query_file: str, qrels_file: str=None):

    if qrels_file is None:
        valid_qids = None
        valid_query_num = int(subprocess.check_output(["wc", "-l", query_file]).decode("utf-8").split()[0])
    else:
        valid_qids = set()
        for line in open(qrels_file, 'r', encoding='utf8'):
            qid, _, pid, qrel = line.split()
            if int(qrel) > 0:
                valid_qids.add(qid)
        valid_query_num = len(valid_qids)
    return valid_qids, valid_query_num


def preprocess_queries(args):

    valid_qids, valid_query_num = count_valid_queries(args.query_file, args.qrels_file)
    out_query_path = args.output_path
    print('start query file split processing')
    splits_dir_lst, _ = multi_file_process(args, args.threads, 
                                           args.query_file, out_query_path, 
                                           pp.query_preprocessing_fn)

    print('start merging splits')

    token_ids_array = np.memmap(out_query_path + ".memmap", shape=(valid_query_num, args.max_query_length), mode='w+', dtype=np.int32)
    qid2offset = {}
    token_length_array = []

    idx = 0
    for split_dir in splits_dir_lst:
        split_qids_array         = np.memmap(os.path.join(split_dir, "ids.memmap"),       mode='r', dtype=np.int32)
        split_token_ids_array    = np.memmap(os.path.join(split_dir, "token_ids.memmap"), mode='r', dtype=np.int32).reshape(len(split_qids_array), -1)
        split_token_length_array = np.memmap(os.path.join(split_dir, "lengths.memmap"),   mode='r', dtype=np.int32)
        
        for qid, token_ids, length in zip(split_qids_array, split_token_ids_array, split_token_length_array):
            if valid_qids is not None and qid not in valid_qids:
                # exclude the query as it has no positive qrels
                continue
            token_ids_array[idx, :] = token_ids
            token_length_array.append(length) 
            qid2offset[qid] = idx
            idx += 1
    assert len(token_length_array) == len(token_ids_array) == idx

    np.save(out_query_path + "_length.npy", np.array(token_length_array))
    pp.dump_json(out_query_path + "_meta.json", idx, args.max_query_length)

    print("Total lines written: " + str(idx))
    
    qid2offset_path = args.output_path + '_qid2offset.pickle'
    with open(qid2offset_path, 'wb') as handle:
        pickle.dump(qid2offset, handle, protocol=4)

    for split_dir in splits_dir_lst:
        shutil.rmtree(split_dir)

    if args.qrels_file is None:
        print("No qrels file provided")
        return
    
    print("Writing qrels")
    with open(os.path.join(args.output_path, 'preprocessed_qrels'), "w", encoding='utf-8') as qrel_output: 
        out_line_count = 0
        for line in open(args.qrels_file, 'r', encoding='utf8'):
            qid, _, pid, rel = line.split()
            qid = int(qid)
            if pid[0].isdigit():
                pid = int(pid)
            else:
                pid = int(pid[1:])
            qrel_output.write(f"{qid2offset[qid]}\t0\t{args.pid2offset[pid]}\t{rel}\n")
            out_line_count += 1
        print("Total lines written: " + str(out_line_count))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--query_file",         required=True,          type=str)
    parser.add_argument("--qrels_file",         default=None,           type=str)
    parser.add_argument("--pid2offset_file",    required=True,          type=str)

    parser.add_argument("--output_path",        required=True,          type=str)
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str)
    parser.add_argument("--max_query_length",   default=64,             type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--threads", default=multiprocessing.cpu_count(), type=int)

    args = parser.parse_args()

    if os.path.exists(args.output_path + '.memmap'):
        print(f"preprocessed data already exist in {args.output_path}, exit preprocessing")
    else:
        preprocess_queries(args)
