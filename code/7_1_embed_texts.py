# fast_embed_minilm_ids_manual_pad.py
# Embeds long texts using sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
# - Splits texts into <=512-token chunks (with overlap)
# - Manual clamp + pad of token IDs (no HF warnings)
# - Big inner chunk batches on GPU (H100-friendly)
# - BF16 + TF32 for speed
# - Weighted mean over chunks per doc, then L2-normalize
# - Writes one JSONL line per bid: {"bid": "...", "embedding": [...]}


from pathlib import Path
import json
from typing import Iterator, Tuple, List

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# ------------------ Config ------------------
DATA_PATH = Path("/work/data_assigments/DATA/")
MODEL_NAME         = "sentence-transformers/all-mpnet-base-v2" #"sentence-transformers/paraphrase-MiniLM-L12-v2"
INPUT_PATH         = DATA_PATH / "all_english_entries_clean.jsonl"
OUTPUT_PATH        = DATA_PATH / "embeddings_english.jsonl"


DOCS_PER_BATCH     = 512 #256      # raise on H100 if IO permits (e.g., 1024)
MAX_TOKENS         = 512       # model limit
OVERLAP_TOKENS     = 0        # reduce for speed (0..64 reasonable)
CHUNK_BATCH_SIZE   = 1024 #1638     # inner batch of chunks to GPU; 
NORMALIZE_OUTPUT   = True

DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE              = torch.bfloat16  # ideal for H100; fp16 also possible
# --------------------------------------------

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

def stream_entries(path: Path) -> Iterator[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            yield (e.get("text", "") or ""), str(e.get("bid"))


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def batch_docs(it: Iterator[Tuple[str, str]], n: int):
    texts, bids = [], []
    for t, b in it:
        texts.append(t)
        bids.append(b)
        if len(texts) == n:
            yield texts, bids
            texts, bids = [], []
    if texts:
        yield texts, bids


def chunk_to_ids(tokenizer, text: str, chunk_tokens: int, overlap_tokens: int) -> List[List[int]]:
    """Split a long text into token-id windows of length <= chunk_tokens (no specials)."""
    ids = tokenizer.encode(text, add_special_tokens=False,truncation=False)
    if not ids:
        return [[]]
    stride = max(1, chunk_tokens - max(0, overlap_tokens))
    chunks = []
    for start in range(0, len(ids), stride):
        piece = ids[start:start + chunk_tokens]
        if not piece:
            break
        chunks.append(piece)
        if start + chunk_tokens >= len(ids):
            break
    return chunks


def build_batch_from_ids(
    tokenizer,
    id_lists: List[List[int]],
    max_len: int,
    add_special_tokens: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Manually clamp and pad token ID lists to max_len. Optionally add [CLS] ... [SEP].
    Returns (input_ids [B,T], attention_mask [B,T]) on CPU (pin before moving to GPU).
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else pad_id
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else pad_id

    special = 2 if add_special_tokens else 0
    clamped = []
    for ids in id_lists:
        if add_special_tokens:
            keep = max_len - special
            seq = ids[:keep]
            seq = [cls_id] + seq + [sep_id]
        else:
            seq = ids[:max_len]
        clamped.append(seq)

    tgt_len = min(max_len, max((len(x) for x in clamped), default=1))
    batch_ids = torch.full((len(clamped), tgt_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((len(clamped), tgt_len), dtype=torch.long)

    for i, ids in enumerate(clamped):
        L = min(len(ids), tgt_len)
        if L > 0:
            batch_ids[i, :L] = torch.tensor(ids[:L], dtype=torch.long)
            attn_mask[i, :L] = 1

    return batch_ids, attn_mask


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1)  # [B,T,1]
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, p=2, dim=-1)

def main():
    print(f"Device: {DEVICE}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModel.from_pretrained(MODEL_NAME)
    mdl.eval().to(DEVICE)
    mdl.to(dtype=DTYPE)  # move params to bf16

    # fresh output
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    num_docs = count_lines(INPUT_PATH)
    total_batches = (num_docs + DOCS_PER_BATCH - 1) // DOCS_PER_BATCH

    with OUTPUT_PATH.open("w", encoding="utf-8") as fout:
        for texts, bids in tqdm(batch_docs(stream_entries(INPUT_PATH), DOCS_PER_BATCH),
                                total=total_batches, desc="Embedding docs (fast)"):
            # Build chunk-id lists per doc
            doc_chunk_ids: List[List[int]] = []
            doc_index: List[int] = []      # chunk -> doc idx
            chunk_weights: List[int] = []  # weight by token length pre-specials

            for i, t in enumerate(texts):
                pieces = chunk_to_ids(tok, t, MAX_TOKENS, OVERLAP_TOKENS)
                if not pieces:
                    pieces = [[]]
                for ids in pieces:
                    doc_chunk_ids.append(ids)
                    doc_index.append(i)
                    chunk_weights.append(max(1, len(ids)))

            if not doc_chunk_ids:
                # unlikely, but handle
                dim = mdl.config.hidden_size
                zero = [0.0] * dim
                for bid in bids:
                    fout.write(json.dumps({"bid": bid, "embedding": zero}, ensure_ascii=False) + "\n")
                continue

            doc_index_t = torch.tensor(doc_index, device=DEVICE, dtype=torch.long)
            weights_t = torch.tensor(chunk_weights, device=DEVICE, dtype=DTYPE)

            dim = mdl.config.hidden_size  # should be 384 for this model
            sums = torch.zeros(len(bids), dim, device=DEVICE, dtype=DTYPE)
            wsum = torch.zeros(len(bids), device=DEVICE, dtype=DTYPE)

            # Process chunks in large inner batches
            for start in range(0, len(doc_chunk_ids), CHUNK_BATCH_SIZE):
                sub_ids = doc_chunk_ids[start:start + CHUNK_BATCH_SIZE]
                sub_doc_idx = doc_index_t[start:start + CHUNK_BATCH_SIZE]
                sub_wts = weights_t[start:start + CHUNK_BATCH_SIZE]

                input_ids, attn = build_batch_from_ids(tok, sub_ids, MAX_TOKENS, add_special_tokens=True)
                # pin + non_blocking transfer
                input_ids = input_ids.pin_memory().to(DEVICE, non_blocking=True)
                attn = attn.pin_memory().to(DEVICE, non_blocking=True)

                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=DTYPE):
                        out = mdl(input_ids=input_ids, attention_mask=attn)
                        sent = mean_pool(out.last_hidden_state, attn).to(DTYPE)  # [B, dim]

                # GPU-side weighted accumulation per doc
                weighted = sent * sub_wts.unsqueeze(1)
                sums.index_add_(0, sub_doc_idx, weighted)
                wsum.index_add_(0, sub_doc_idx, sub_wts)

            embs = sums / wsum.clamp(min=1e-6).unsqueeze(1)
            if NORMALIZE_OUTPUT:
                embs = l2norm(embs)

            # write
            embs = embs.float().cpu().numpy().astype("float32")
            for bid, vec in zip(bids, embs):
                fout.write(json.dumps({"bid": bid, "embedding": vec.tolist()}, ensure_ascii=False) + "\n")

    print(f"Done. Saved embeddings to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()