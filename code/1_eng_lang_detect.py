#!/usr/bin/env python
# coding: utf-8

import json
from tqdm import tqdm
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0


def detect_middle(text, num_words=500):
    words = text.split()
    if not words:
        return "unknown"
    mid = len(words) // 2
    half = num_words // 2
    start = max(0, mid - half)
    end = mid + half
    snippet = " ".join(words[start:end]).strip()
    if not snippet:
        return "unknown"
    try:
        return detect(snippet)
    except LangDetectException:
        return "unknown"


def find_english(input_jsonl, danish_bids, english_output_path):
    danish_set = set(danish_bids)
    english_bids = []

    print(f"📂 Reading entries from: {input_jsonl}")
    print(f"📝 Writing English entries to: {english_output_path}")
    print(f"🔍 Skipping {len(danish_set)} known Danish BIDs...")

    # Pre-count non-Danish lines for a precise progress bar total
    print("🧮 Counting non-Danish entries for progress bar...")
    with open(input_jsonl, "r", encoding="utf-8") as f:
        total_non_danish = sum(1 for line in f if json.loads(line)["bid"] not in danish_set)
    print(f"➡️  Will scan {total_non_danish} non-Danish entries.")

    with open(input_jsonl, "r", encoding="utf-8") as f_in, \
         open(english_output_path, "w", encoding="utf-8") as f_out, \
         tqdm(total=total_non_danish, desc="Finding English among non-Danish") as pbar:

        for line in f_in:
            obj = json.loads(line)
            bid = obj["bid"]
            if bid in danish_set:
                continue  # not counted toward the bar

            text = obj.get("text", "")
            # Skip if no text / whitespace-only text
            if not text or not text.strip():
                pbar.update(1)
                continue

            lang = detect_middle(text)
            if lang == "en":
                english_bids.append(bid)
                f_out.write(json.dumps({"bid": bid, "text": text}, ensure_ascii=False) + "\n")

            pbar.update(1)

    print(f"✅ Finished. Found {len(english_bids)} English entries.")
    return english_bids


# --- Example usage ---
if __name__ == "__main__":
    input_jsonl = "/work/data_assigments/DATA/all_entries.jsonl"
    danish_jsonl = "/work/data_assigments/DATA/all_danish_entries.jsonl"
    english_output = "/work/data_assigments/DATA/all_english_entries.jsonl"

    print("📄 Loading Danish BIDs...")
    with open(danish_jsonl, "r", encoding="utf-8") as f:
        danish_bids = [json.loads(line)["bid"] for line in f]
    print(f"✅ Loaded {len(danish_bids)} Danish BIDs.")

    english_bids = find_english(input_jsonl, danish_bids, english_output)

    print(f"✨ Script completed. Total English BIDs collected: {len(english_bids)}")
