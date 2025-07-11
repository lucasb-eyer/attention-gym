import time

import torch
from torch.nn.attention.flex_attention import create_block_mask

from attn_gym.masks import causal_mask, generate_doc_mask_mod
from attn_gym.masks.document_mask import generate_random_lengths, length_to_offsets


def med(x):
    return torch.median(torch.tensor(x))


def main():
    torch.random.manual_seed(123)
    device = torch.device("cuda")

    cbm = torch.compile(create_block_mask)

    for seqlen in (128, 512, 1024, 4096, 16384, 32768, 65536, 131072, 262144, 524288, 1048576):
        for ndoc in (4, 16, 64, 128, 1024):
            if ndoc >= seqlen:
                continue

            print(f"Benchmarking {seqlen=} {ndoc=}...", flush=True, end="")
            lengths = generate_random_lengths(total_length=seqlen, num_documents=ndoc)
            offsets = length_to_offsets(lengths, device)
            mask_mod_fn = generate_doc_mask_mod(mask_mod=causal_mask, offsets=offsets)

            seconds, peakbytes = [], []
            for _ in range(10):
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                cbm(mask_mod_fn, B=None, H=None, Q_LEN=seqlen, KV_LEN=seqlen, device=device)
                torch.cuda.synchronize(device)
                seconds.append(time.perf_counter() - t0)
                peakbytes.append(torch.cuda.max_memory_allocated())
            print(
                f" median t={med(seconds)*1000:.2f}ms, max m={max(peakbytes) / 1024 / 1024:.2f}MiB",
                flush=True,
            )


if __name__ == "__main__":
    main()
