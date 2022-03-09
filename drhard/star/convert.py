# coding=utf-8
import os
import argparse
import subprocess
import numpy as np
import torch

from tqdm import tqdm
from pathlib import Path
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
from transformers import RobertaConfig

from model import RobertaDot
from data.dataset import TokenIdsCache
from data.dataset import SequenceDataset
from data.dataset import single_get_collate_function


def prediction(model, data_collator, args, dataset, embedding_memmap, ids_memmap, is_query):
   
    dataloader = DataLoader(dataset,
        sampler=SequentialSampler(dataset),
        batch_size=args.batch_size * args.n_gpu,
        collate_fn=data_collator,
        drop_last=False,
    )

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    print("***** Running *****")
    print(f"  Num examples = {dataloader.batch_size}")
    print(f"  Batch size = {len(dataloader.dataset)}")

    model.eval()
    write_index = 0
    for inputs, ids in tqdm(dataloader):
        for k, v in inputs.items():
            inputs[k] = v.to(args.device)

        logits = model(is_query=is_query, **inputs).detach().cpu().numpy()
        write_size = len(logits)
        assert write_size == len(ids)
        embedding_memmap[write_index:write_index + write_size] = logits
        ids_memmap[write_index:write_index + write_size] = ids
        write_index += write_size
    assert write_index == len(embedding_memmap) == len(ids_memmap)


def inference(model, args, embedding_size):

    max_length = args.max_query_length if args.is_query else args.max_doc_length
    
    collator = single_get_collate_function(max_length)
    ids_cache = TokenIdsCache(args.input_prefix)
    dataset = SequenceDataset(ids_cache=ids_cache, max_seq_length=max_length)

    emb_memmap = np.memmap(args.emb_memmap_path, dtype=np.float32, mode="w+", shape=(len(dataset), embedding_size))
    ids_memmap = np.memmap(args.ids_memmap_path, dtype=np.int32,   mode="w+", shape=(len(dataset), ))

    try:
        prediction(model, collator, args, dataset, emb_memmap, ids_memmap, is_query=args.is_query)
    except:
        subprocess.check_call(["rm", args.emb_memmap_path])
        subprocess.check_call(["rm", args.ids_memmap_path])
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str)
    parser.add_argument("--input_file",  required=True,                 type=str)
    parser.add_argument("--output_path", required=True,                 type=str)
    parser.add_argument("--max_query_length", type=int, default=32)
    parser.add_argument("--max_doc_length",   type=int, default=512)
    parser.add_argument("--batch_size",       type=int, default=32)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--is_query", action="store_true")
    args = parser.parse_args()

    # Computed arguments
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.output_prefix = os.path.join(args.output_path, Path(args.input_file).stem)

    args.emb_memmap_path    = args.output_prefix + ".emb.memmap"
    args.ids_memmap_path = os.path.join(args.output_path, ".ids.memmap")

    config = RobertaConfig.from_pretrained(args.model_path, gradient_checkpointing=False)
    model = RobertaDot.from_pretrained(args.model_path, config=config)
    model = model.to(args.device)
    output_embedding_size = model.output_embedding_size

    if os.path.exists(args.emb_memmap_path):
        print(f"converted data already exist in {args.output_path}, exit conversion")
    else:
        os.makedirs(args.output_path, exist_ok=True)
        inference(model, args, output_embedding_size)

if __name__ == "__main__":
    main()
