import os
import pickle
import argparse

from pathlib import Path


def preprocess(args):

    pid2offset = pickle.load(open(args.pid2offset_file, 'rb'))
    qid2offset = pickle.load(open(args.qid2offset_file, 'rb'))

    print("Writing qrels")
    with open(args.output_prefix + '.preproc', 'wt', encoding='utf-8') as qrel_output: 
        out_line_count = 0
        for line in open(args.qrels_file, 'r', encoding='utf8'):
            qid, _, pid, rel = line.split()
            qid = int(qid)
            if pid[0].isdigit():
                pid = int(pid)
            else:
                pid = int(pid[1:])
            qrel_output.write(f"{qid2offset[qid]}\t0\t{pid2offset[pid]}\t{rel}\n")
            out_line_count += 1
        print("Total lines written: " + str(out_line_count))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--qrels_file",         default=None,           type=str)
    parser.add_argument("--qid2offset_file",    required=True,          type=str)    
    parser.add_argument("--pid2offset_file",    required=True,          type=str)

    parser.add_argument("--output_path",        required=True,          type=str)

    args = parser.parse_args()

    args.output_prefix = os.path.join(args.output_path, Path(args.qrels_file).stem)

    if os.path.exists(args.output_prefix + ".preproc"):
        print(f"preprocessed data already exist in {args.output_path}, exit preprocessing")
    else:
        os.makedirs(args.output_path, exist_ok=True)
        preprocess(args)