import os
import torch
import argparse
import torch
import faiss

import numpy as np


def construct_flatindex_from_embeddings(embeddings, ids=None):
    dim = embeddings.shape[1]
    # print('embedding shape: ' + str(embeddings.shape))
    # index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    index = faiss.IndexFlatIP(dim)
    if ids is not None:
        ids = ids.astype(np.int64)
        # print(ids.shape, ids.dtype)
        index = faiss.IndexIDMap2(index)
        index.add_with_ids(embeddings, ids)
    else:
        index.add(embeddings)
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--de", type=str, required=True)
    parser.add_argument("--di", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    if os.path.exists(args.out):
        print(f"indexed data already exist in {args.out}, exit indexing")
    else:
        os.makedirs(args.out, exist_ok=True)
        doc_ids        = np.memmap(args.di, dtype=np.int32,   mode="r")
        doc_embeddings = np.memmap(args.de, dtype=np.float32, mode="r")
        doc_embeddings = doc_embeddings.reshape(len(doc_ids), -1)

        index = construct_flatindex_from_embeddings(doc_embeddings, doc_ids)

        faiss.write_index(index, args.out)


if __name__ == "__main__":
    main()