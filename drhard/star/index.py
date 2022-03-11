import os
import torch
import argparse
import torch
import faiss

import numpy as np

from tqdm import tqdm
from pathlib import Path

def construct_flatindex_from_embeddings(embeddings, ids=None, bs=32):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    if ids is not None:
        ids = ids.astype(np.int64)
        index = faiss.IndexIDMap2(index)
        for batchpos in tqdm(range(0, len(embeddings), bs)):
            endpos = min(batchpos + bs, len(embeddings))
            index.add_with_ids(embeddings[batchpos:endpos], ids[batchpos:endpos])
    else:
        for batchpos in tqdm(range(0, len(embeddings), bs)):
            endpos = min(batchpos + bs, len(embeddings))
            index.add(embeddings[batchpos:endpos])
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--de", type=str, required=True)
    parser.add_argument("--di", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--bs",  type=int, default=1024)
    args = parser.parse_args()

    if os.path.exists(args.out):
        print(f"indexed data already exist: {args.out}, exit indexing")
    else:
        os.makedirs(Path(args.out).parent, exist_ok=True)
        doc_ids        = np.memmap(args.di, dtype=np.int32,   mode="r")
        doc_embeddings = np.memmap(args.de, dtype=np.float32, mode="r")
        doc_embeddings = doc_embeddings.reshape(len(doc_ids), -1)

        index = construct_flatindex_from_embeddings(doc_embeddings, doc_ids, args.bs)

        faiss.write_index(index, args.out)


if __name__ == "__main__":
    main()