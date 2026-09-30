from pathlib import Path
import json
from tqdm import tqdm
import spacy
from spacy.matcher import PhraseMatcher
from collections import Counter

# -----------------------------
# Config
# -----------------------------
DATA_PATH = Path("/work/data_assigments/DATA/")
COMMON_PATH = DATA_PATH / "common_LLM_words.txt"
RARE_PATH   = DATA_PATH / "rare_LLM_words.txt"
JSONL_PATH = DATA_PATH / "all_english_entries_clean.jsonl"
output_path = DATA_PATH / "bid2LLMWordMeasuresEnglish.jsonl"
# -----------------------------
# Helpers
# -----------------------------
def load_markers_csv_lines(path: Path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("\n", 1)[0].strip()
            if not line:
                continue
            for word in line.split(","):
                word = word.strip()
                if word:
                    out.append(word)
    if not out:
        raise RuntimeError(f"No markers found in {path}")
    return out

def BuildPhraseMatcher(nlp, common_markers, rare_markers):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    matcher.add("Commonlist", [nlp.make_doc(word) for word in common_markers])
    matcher.add("Rarelist",   [nlp.make_doc(word) for word in rare_markers])
    return matcher
    
def CountMatches(text, matcher):
    doc = nlp.make_doc(text)
    word_tokens = sum(1 for t in doc if t.is_alpha)

    matches = matcher(doc)

    common_counter = Counter()
    rare_counter   = Counter()

    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        span_text = doc[start:end].text.lower()
        if label == "Commonlist":
            common_counter[span_text] += 1
        elif label == "Rarelist":
            rare_counter[span_text] += 1

    # summary stats
    common_count = sum(common_counter.values())
    rare_count   = sum(rare_counter.values())

    return {
        "common_count": common_count,
        "rare_count": rare_count,
        "common_unique": len(common_counter),
        "rare_unique": len(rare_counter),
        "common_freqs": dict(common_counter),
        "rare_freqs": dict(rare_counter),
        "n_words": word_tokens,
    }

# -----------------------------
# spaCy setup
# -----------------------------
nlp = spacy.blank("en")
nlp.max_length = 5_000_000

# -----------------------------
# Count valid observations first
# -----------------------------
obs_count = 0
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        txt = obj.get("text", "")
        if txt and txt.strip():
            obs_count += 1

print(f"Found {obs_count:,} valid observations in {JSONL_PATH}")


# -----------------------------
# Load markers & build matcher
# -----------------------------
common_markers = load_markers_csv_lines(COMMON_PATH)
rare_markers   = load_markers_csv_lines(RARE_PATH)

print(f"Loaded {len(common_markers)} common markers and {len(rare_markers)} rare markers.")

matcher = BuildPhraseMatcher(nlp, common_markers, rare_markers)

# -----------------------------
# Process JSONL
# -----------------------------
with open(JSONL_PATH, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:

    for line in tqdm(fin, total=obs_count):
        if not line.strip():
            continue
        obj = json.loads(line)
        txt = obj.get("text", "")
        if txt and txt.strip():
            bid = obj.get("bid")
            stats = CountMatches(txt, matcher)

            # write directly to output file
            entry = {"bid": bid, **stats}
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")