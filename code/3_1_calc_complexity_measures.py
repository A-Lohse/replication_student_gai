#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os
import sys
from tqdm import tqdm
import spacy
from lexical_diversity import lex_div as ld
import math
import pyphen
import gc

# === CLI setup ===
parser = argparse.ArgumentParser(description="Compute writing quality measures from Danish student texts.")
parser.add_argument("--rerun", action="store_true", help="Force full recomputation of all entries (ignore existing output).")
parser.add_argument("--resume", action="store_true", help="Resume processing: append to existing output and skip already-processed BIDs.")
args = parser.parse_args()

# === Config ===
input_path = '/work/data_assigments/DATA/all_english_entries_clean.jsonl'
output_path = '/work/data_assigments/DATA/bid2WritingQualityMeasuresEnglish.jsonl'
skipped_bids_path = output_path.replace(".jsonl", "_skipped.jsonl")
batch_size = 32


# Helper functions
def try_measure(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        return None

def cttr(tokens):
    types = len(set(tokens))
    tokens_len = len(tokens)
    return types / (2 * tokens_len) ** 0.5 if tokens_len > 0 else None

def get_depth(token):
    if not list(token.children):
        return 1
    return 1 + max(get_depth(child) for child in token.children)

def get_max_tree_depths(doc):
    depths = [get_depth(sent.root) for sent in doc.sents]
    if not depths:
        return None, None
    return sum(depths) / len(depths), max(depths)

def get_mean_tunit_length(doc):
    t_units = []
    for sent in doc.sents:
        roots = [token for token in sent if token.dep_ in ("ROOT", "conj") and token.pos_ == "VERB"]
        for root in roots:
            t_units.append(len(list(root.subtree)))
    return sum(t_units) / len(t_units) if t_units else None

def count_clauses(doc):
    finite_verbs = [
        token for token in doc
        if token.pos_ == "VERB" and "Fin" in token.morph.get("VerbForm")
    ]
    return len(finite_verbs)

# Danish hyphenator for syllables (for Flesch-Kincaid)
dic = pyphen.Pyphen(lang='en_US')

def count_syllables(word):
    parts = dic.inserted(word)
    return parts.count('-') + 1 if parts else 1

# Path to Dale–Chall easy word list (English, one word per line, lowercase)
DALE_CHALL_PATH = os.path.join("/work/data_assigments/DATA", "dale_chall_word_list.txt")

def load_easy_words_en(path=DALE_CHALL_PATH):
    """
    Load the New Dale–Chall easy word list (one word per line, lowercase).
    """
    easy = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip().lower()
                if w:
                    easy.add(w)
    except FileNotFoundError:
        raise RuntimeError(f"❌ Dale–Chall word list not found at {path}")
    return easy

EASY_WORDS_EN = load_easy_words_en()

def dale_chall_score_en(words, num_sentences, easy_words=EASY_WORDS_EN):
    """
    Compute English Dale–Chall readability score.
    """
    num_words = len(words)
    if num_words == 0 or num_sentences == 0:
        return None

    difficult = sum(1 for w in words if w.lower() not in easy_words)
    pct_difficult = difficult / num_words

    raw = 0.1579 * (pct_difficult * 100.0) + 0.0496 * (num_words / num_sentences)
    if pct_difficult <= 0.05:
        raw -= 3.6365
    return raw


def compute_readability(doc):
    words = [token.text for token in doc if token.is_alpha]
    lemmas = [token.lemma_.lower() for token in doc if token.is_alpha]

    num_words = len(words)
    num_chars = sum(len(w) for w in words)
    num_sentences = sum(1 for _ in doc.sents)
    num_syllables = sum(count_syllables(w) for w in words)
    long_words = sum(1 for w in words if len(w) > 6)
    vocab = set(lemmas)

    if num_words == 0 or num_sentences == 0:
        return {
            "num_words": None,
            "num_chars": None,
            "num_sentences": None,
            "num_syllables": None,
            "ARI": None,
            "chars_per_word": None,
            "flesch_kincaid": None,
            "lix": None,
            "rix": None,
            "ovix": None,
            "dale_chall": None
        }

    ari = 4.71 * (num_chars / num_words) + 0.5 * (num_words / num_sentences) - 21.43
    fk = 0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59
    lix = (num_words / num_sentences) + (long_words * 100.0 / num_words)
    rix = long_words / num_sentences

    try:
        ovix = math.log(num_words) / (math.log(2) - math.log(len(vocab)))
    except Exception:
        ovix = None

    dc = dale_chall_score_en(words, num_sentences, EASY_WORDS_EN)

    return {
        "num_words": num_words,
        "num_chars": num_chars,
        "num_sentences": num_sentences,
        "num_syllables": num_syllables,
        "ARI": ari,
        "chars_per_word": num_chars / num_words,
        "flesch_kincaid": fk,
        "lix": lix,
        "rix": rix,
        "ovix": ovix,
        "dale_chall": dc
    }


def process_batch(batch):
    
    texts = [item['text'] for item in batch]
    bids = [item['bid'] for item in batch]
    docs = list(nlp.pipe(texts, batch_size=batch_size))

    results = []
    for bid, doc in zip(bids, docs):
        tokens = [token.lemma_.lower() for token in doc if token.is_alpha]

        # Lexical diversity measures
        lexical = {
            "TTR": try_measure(ld.ttr, tokens),
            "Root TTR": try_measure(ld.root_ttr, tokens),
            "Log TTR": try_measure(ld.log_ttr, tokens),
            "Maas": try_measure(ld.maas_ttr, tokens),
            "MSTTR": try_measure(ld.msttr, tokens, window_length=50),
            "MATTR": try_measure(ld.mattr, tokens, window_length=50),
            "HDD": try_measure(ld.hdd, tokens),
            "MTLD": try_measure(ld.mtld, tokens),
            "CTTR": try_measure(cttr, tokens),
        }

        # Syntactic complexity measures
        mean_tree_depth, max_tree_depth = try_measure(get_max_tree_depths, doc) or (None, None)
        clause_count = try_measure(count_clauses, doc)
        mean_tunit_length = try_measure(get_mean_tunit_length, doc)

        # Readability measures
        readability = try_measure(compute_readability, doc) or {
            "ARI": None,
            "chars_per_word": None,
            "flesch_kincaid": None,
            "lix": None,
            "rix": None,
            "ovix": None,
            "dale_chall": None
        }

        # Final result with all metrics
        results.append({
            bid: {
                **lexical,
                "mean_tree_depth": mean_tree_depth,
                "max_tree_depth": max_tree_depth,
                "clause_count": clause_count,
                "mean_tunit_length": mean_tunit_length,
                **readability,
            }
        })

    return results


# === Determine run mode and handle existing output ===
if args.rerun and args.resume:
    print("❌ You cannot use both --rerun and --resume at the same time.")
    sys.exit(1)

if not args.rerun and not args.resume:
    print("❌ You must specify either --rerun (overwrite and restart) or --resume (append and skip).\nNow just reading and counting existing entries.")
    # Step 0: Load existing BIDs from output file if it exists
    existing_bids = set()
    if os.path.exists(output_path):
        print("🔄 Reading existing BIDs from output file...")
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    bid = next(iter(obj))  # assumes {bid: {...}}
                    existing_bids.add(bid)
                except Exception:
                    continue
        print(f"✅ Found {len(existing_bids)} entries already processed.\n")

    sys.exit(1)

if args.resume:
    print("🔄 Resume mode: reading existing BIDs from output file...")
    if os.path.exists(output_path):
        existing_bids = set()
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    bid = next(iter(obj))  # assumes {bid: {...}}
                    existing_bids.add(bid)
                except Exception:
                    continue
        print(f"✅ Found {len(existing_bids)} entries already processed.\n")
    else:
        print("⚠️ Output file not found. Starting fresh.")
    mode = 'a'

elif args.rerun:
    existing_bids = set() #nothing in it
    print("🔁 Full rerun requested. Overwriting output file and recomputing all entries.")
    mode = 'w'




# Step 1: Count lines in the input file (total BIDs to process)
with open(input_path, 'r', encoding='utf-8') as f:
    total_entries = 0
    remaining_entries = 0
    for line in f:
        try:
            obj = json.loads(line)
            bid = obj["bid"]
            total_entries += 1
            
            if bid not in existing_bids:
                remaining_entries += 1
        except Exception:
            continue

# Step 2: Calculate number of batches still to process
total_batches = math.ceil(total_entries / batch_size)
remaining_batches = math.ceil(remaining_entries / batch_size)

print(f"📦 Found {total_entries} total entries.")
print(f"⏭️ Skipping {total_entries - remaining_entries} already processed entries.")
print(f"🚀 Will process approximately {remaining_batches} new batches out of {total_batches} total batches of size {batch_size}")




# Load spaCy Danish model with sentence segmentation enabled
print("🔄 Loading spaCy model...")
nlp = spacy.load("en_core_web_sm", disable=["ner"])
nlp.max_length = 2_000_000  # ⬅️ This is the new line you add
print("✅ spaCy model loaded.\n")


skipped_bids = []
with open(input_path, 'r', encoding='utf-8') as fin, \
     open(output_path, mode, encoding='utf-8') as fout, \
     open(skipped_bids_path, 'a', encoding='utf-8') as fskip:  # log skipped entries

    buffer = []
    batch_count = 0

    # Create a batch progress bar
    pbar = tqdm(total=total_batches, desc="Processing batches")
    pbar.update(total_batches - remaining_batches)

    for line in fin:
        obj = json.loads(line)
        bid = obj["bid"]
        text = obj["text"]

        # Skip if already processed
        if bid in existing_bids:
            continue

        # Skip if text too long for spaCy
        if len(text) > nlp.max_length:
            skipped_bids.append(bid)
            fskip.write(json.dumps({"bid": bid, "reason": "text too long"}) + "\n")
            continue

        buffer.append({"bid": bid, "text": text})

        if len(buffer) == batch_size:
            batch_results = process_batch(buffer)
            for item in batch_results:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            buffer = []
            batch_count += 1
            pbar.update(1)
            gc.collect()

    # Final batch
    if buffer:
        batch_results = process_batch(buffer)
        for item in batch_results:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        batch_count += 1
        pbar.update(1)

    pbar.close()


# === Second pass: Try processing skipped BIDs one by one ===
print("\n🔁 Reprocessing skipped BIDs one by one with increased max_length...")

nlp.max_length = 5000000  # Increase spaCy document length limit

if os.path.exists(skipped_bids_path):
    with open(skipped_bids_path, 'r', encoding='utf-8') as f:
        skipped_bids_set = {json.loads(line)["bid"] for line in f if "bid" in line}

    recovered_count = 0

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'a', encoding='utf-8') as fout:

        for line in tqdm(fin, desc="Reprocessing skipped BIDs"):
            try:
                obj = json.loads(line)
                bid = obj["bid"]
                if bid not in skipped_bids_set:
                    continue
                text = obj["text"]

                result = process_batch([{"bid": bid, "text": text}])
                for item in result:
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")

                recovered_count += 1
                skipped_bids_set.remove(bid)

            except Exception as e:
                raise RuntimeError(f"❌ Failed to process skipped BID {bid}: {e}")

    if skipped_bids_set:
        raise RuntimeError(f"❌ Some skipped BIDs were not found in the input file: {skipped_bids_set}")

    os.remove(skipped_bids_path)
    print(f"✅ Successfully reprocessed all previously skipped entries ({recovered_count}).")
else:
    print("✅ No skipped file found. Skipping second pass.")
