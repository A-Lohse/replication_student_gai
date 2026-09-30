#!/usr/bin/env python
# coding: utf-8

"""
English-adapted version of the text cleaning pipeline.
Changes made:
- Uses NLTK 'punkt_tab' and sets sentence tokenization language to English.
- Loads spaCy English model (en_core_web_sm).
- Replaces Danish keyword lists and regex patterns with English equivalents.
- Replaces Danish abbreviation cleaner with an English version.
- Generalizes phone number replacement (no longer Danish +45 specific).
- (Optional) Input/output paths now point to English entries.
"""

import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
import re
from collections import Counter
import random
from tqdm import tqdm
import string

import spacy
from collections import defaultdict
from rapidfuzz import fuzz
import nltk
from nltk.tokenize import sent_tokenize

# === CLI setup ===
parser = argparse.ArgumentParser(description="Clean texts again, resume, or load from memory.")
parser.add_argument("--rerun", action="store_true", help="Force rerun of text cleaning (start from scratch)")
parser.add_argument("--resume", action="store_true", help="Resume cleaning by skipping already-processed entries")
args = parser.parse_args()


# === Path setup ===
# Point to your English entries; change as needed.
input_path =  '/work/data_assigments/DATA/all_english_entries.jsonl'
output_path = '/work/data_assigments/DATA/all_english_entries_clean.jsonl'

cstats_file = '/work/data_assigments/DATA/eng_c_parts.json'
cparts_file = '/work/data_assigments/DATA/eng_c_stats.parquet'


# === Check for existing output ===
if not args.rerun and not args.resume:
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as fout:
                count = sum(1 for _ in fout)
            print(f"⏭️  Not rerunning or resuming. Existing file has {count} entries. Use --rerun or --resume.")
            sys.exit(0)
        except Exception as e:
            print(f"⚠️  Could not read existing output. Error: {e}")
            sys.exit(0)
    else:
        print("⚠️  Output file not found. Proceeding with computation...")


# Download the tokenizer model if not already present (newer NLTK)
nltk.download('punkt_tab')
# Load English spaCy model
nlp = spacy.load("en_core_web_sm", disable=["ner"])  #"parser", "lemmatizer", "attribute_ruler", "textcat"])

# === Keyword lists (English) ===
intro_keywords = ["introduction", "background", "overview", "problem statement", "motivation"]
backup_keywords = ["chapter"]
toc_head_keywords = ["table of contents", "contents", "overview"]
literature_keywords = ["references", "bibliography", "works cited", "sources", "literature"]
bilag_keywords = ["appendix", "appendices"]
section_prefixes = ["appendix", "chapter", "section"]
page_keywords = ["page"]
page_connectors = ["of"]

# Pre-compile patterns with keyword lists
COMPILED_PATTERNS = {
    'tab_spaces': re.compile(r'\t\r\s\xa0'),
    'multiple_newlines': re.compile(r'\n\s*\n\s*\n+'),

    'intro_pattern': re.compile(
        rf'^\s*'
        rf'(?:chapter\s+\d+(?:\.\d+)*\s*[-–—:\.]?\s*|\d+(?:\.\d+)*\s*[-–—:\.]?\s*)?'
        rf'("|".join(intro_keywords))'
        rf'\s*[:\-–—\.]?'
        rf'(?:\s*\([^)]+\))?'
        rf'\s*$',
        re.IGNORECASE
    ),

    'backup_pattern': re.compile(
        rf'^\s*(chapter\s+1(?:\.0|\.1)?|1\.?(:?\.0|\.1)?\s*chapter)\s*$',
        re.IGNORECASE
    ),

    # English letters only
    'toc_pattern': re.compile(r'^\s*\d+(?:\.\d+)*\s+[A-Za-z].*$'),

    'toc_head_pattern': re.compile(
        rf'^\s*\b({"|".join(toc_head_keywords)})\b\s*$',
        re.IGNORECASE
    ),

    'literature_pattern': re.compile(
        rf'^\s*(?:'
        rf'(?:{"|".join(section_prefixes)})\s+\d+(?:\.\d+)*\s*[:\-–—\.]?\s*|'
        rf'\d+(?:\.\d+)*\s*[:\-–—\.]?\s*)?'
        rf'({"|".join(literature_keywords)})\s*[:\-–—\.\']?\s*$',
        re.IGNORECASE
    ),

    'bilag_pattern': re.compile(
        rf'^\s*(?:'
        rf'(?:chapter|section)\s+\d+(?:\.\d+)*\s*[:\-–—\.]?\s*|'
        rf'\d+(?:\.\d+)*\s*[:\-–—\.]?\s*)?'
        rf'({"|".join(bilag_keywords)})\b\s*1?[:\-–—\.]?\s*$',
        re.IGNORECASE
    ),

    'footnote_pattern': re.compile(r'^\d{1,3}\s+[^\s,]{2,}.*'),

    'page_number_pattern': re.compile(
        rf'^\s*(?:'
        rf'(?:{"|".join(page_keywords)})\s+\d+'
        rf'|(?:{"|".join(page_keywords)})\s+\d+\s+(?:{"|".join(page_connectors)})\s+\d+'
        rf'|\d+\s*/\s*\d+'
        rf'|\d+\s*[-–—]\s*\d+'
        rf'|\d+'
        rf')\s*$',
        re.IGNORECASE
    ),

    'numeric_citations': re.compile(r'\[\s*\d+(\s*[-–,]\s*\d+)*\s*\]'),
    'year_citations': re.compile(r'\([^)]*(\b\d{4}\b|ibid\.?)[^)]*\)', re.IGNORECASE),
    'empty_parens': re.compile(r'\(\s*\)'),
    'multiple_spaces': re.compile(r'\s{3,}'),
    'space_before_punct': re.compile(r'\s+([\.,;:!?])'),
    'double_punct': re.compile(r'([\.,;:!?])\s*([\.,;:!?])'),
    'trailing_space': re.compile(r'\s+$'),
    'leading_numbers': re.compile(r'^\d+\s*'),
    'link_pattern': re.compile(r'\b(?:https?://|www\.)\S+', re.IGNORECASE),
    'url_replace': re.compile(r'https?://\S+|www\.\S+'),
    'email_replace': re.compile(r'\b[\w.+-]+@[\w.-]+\.\w{2,}\b'),
    # Generic international phone pattern
    'phone_replace': re.compile(r'\b(?:\+?\d[\d\-\.\s\(\)]{7,}\d)\b'),
    'long_dashes': re.compile(r'[─\-]{5,}'),
    'hyphen_break': re.compile(r'(?<=\w)-\s+(?=\w)'),
    'tab_spaces_final': re.compile(r'[ \t]+'),
}

tr = str.maketrans('', '', string.punctuation)

# === English abbreviations cleaner ===
def remove_english_abbrev_dots(text: str) -> str:
    repl = {
        "e.g.": "eg", "i.e.": "ie", "et al.": "etal", "etc.": "etc",
        "Fig.": "Fig", "Figs.": "Figs", "Dr.": "Dr", "Mr.": "Mr", "Mrs.": "Mrs", "Ms.": "Ms",
        "vs.": "vs", "cf.": "cf", "approx.": "approx",
        "Jan.": "Jan", "Feb.": "Feb", "Mar.": "Mar", "Apr.": "Apr", "Aug.": "Aug",
        "Sept.": "Sept", "Oct.": "Oct", "Nov.": "Nov", "Dec.": "Dec",
    }
    if not repl:
        return text
    pattern = re.compile(r'(?:(?<=\s)|(?<=^))(' + '|'.join(re.escape(k) for k in repl) + r')')
    return pattern.sub(lambda m: repl[m.group(1)], text)

# === Formatting / cleaning helpers ===

def fmt_check(text, debug=False):
    """
    Detects and fixes unusual formatting issues in text, such as repeated punctuation in place of spaces.
    Applies global and local fixes for common symbols, removes excess whitespace, and returns the cleaned text.
    """

    l = len(text)
    if l <= 500:
        if debug:
            return {"cache": {}, "applied": []}
        return text, []

    d = {
        "plus": "+",
        "excl": "!",
        "dash": "-",
        "pct": "%",
        "s_apos": "'",
        "d_apos": '"',
        "l_paren": "(",
        "r_paren": ")",
    }

    cache = {punc: text.count(val) for punc, val in d.items()}
    applied = []

    format_error_detected = any((cache[punc] / l) > 0.08 for punc in d)

    if format_error_detected:
        for punc, symbol in d.items():
            if (cache[punc] / l) > 0.08:
                text = " ".join(text.split(symbol))
                applied.append(f"global-{punc}")

        lines = text.split("\n")
        c_lines = []
        symbol_set = set(d.values())

        for line in lines:
            stripped = line.strip()
            if stripped in symbol_set:
                continue
            if line.endswith("("):
                line = line.rstrip("(")
            c_line = line
            for punc, symbol in d.items():
                if symbol in line and len(line.split(symbol)) > 6:
                    c_line = " ".join(c_line.split(symbol))
                    applied.append(f"local-{punc}")
            c_lines.append(c_line)
        text = "\n".join(c_lines)

    if '\t\r \xa0' in text:
        text, matches = COMPILED_PATTERNS['tab_spaces'].subn(' ', text)
        applied.append(f"\t\r \xa0 - {matches}")

    l2 = len(text)
    text = COMPILED_PATTERNS['multiple_newlines'].sub('\n\n', text)
    if l2 != len(text):
        applied.append(f"rmv ws - {l2 - len(text)}")

    if debug:
        return {"cache": cache, "applied": list(set(applied))}

    return text, applied


def split_at_intro(text):
    """
    Splits the document at the introduction section if detected by matching known keywords or patterns.
    Returns the text starting from the detected introduction; otherwise, returns the original text.
    """

    lines = text.split('\n')

    if len(lines) < 50:
        return text

    max_index = int(0.1 * len(lines))

    for i in range(max_index):
        line = lines[i]
        cleaned_line = line.replace('\x0c', '').strip()
        if not cleaned_line:
            continue

        match_type = None
        if COMPILED_PATTERNS['intro_pattern'].match(cleaned_line):
            match_type = 'keyworded'
        elif COMPILED_PATTERNS['backup_pattern'].match(cleaned_line):
            match_type = 'backup'

        if match_type:
            for j in range(i + 1, min(i + 3, len(lines))):
                next_line = lines[j].strip()
                if not next_line:
                    continue
                next_line_clean = next_line.translate(tr)
                if len(next_line_clean.split()) >= 8:
                    return '\n'.join(lines[i:]).strip()

    return text


def is_probably_toc_start(lines, start, window=10, thresh=None):
    """
    Heuristically determines whether a given window of lines is the start of a table of contents.
    Uses patterns, short-line density, and dot-density checks.
    """

    if thresh is None:
        thresh = int(window * 0.6)

    end_idx = min(start + window, len(lines))
    slice_lines = lines[start:end_idx]

    if len(slice_lines) == 0:
        return False

    match_count = dot_dense_count = short_count = empty_count = 0

    for line in slice_lines:
        stripped = line.strip()
        if not stripped:
            empty_count += 1
            continue

        if len(line.split()) < 6:
            short_count += 1

        if COMPILED_PATTERNS['toc_pattern'].match(line):
            match_count += 1

        if line.count('.') >= 5:
            dot_dense_count += 1

        if empty_count / window > 0.6:
            return True
        if (dot_dense_count >= thresh and short_count >= thresh):
            return True
        if (match_count >= thresh and short_count >= thresh):
            return True

    return False


def clean_toc(text, max_end=None, max_start=100):
    """
    Detects and removes the table of contents section from the text.
    Uses keyword matches and heuristics to locate the TOC start and end positions.
    """

    lines = text.split('\n')
    line_count = len(lines)
    if line_count < 100 or len(text) < 3000:
        return text, []

    if max_end is None:
        if line_count < 800:
            max_end = 100
        elif line_count >= 800 and line_count < 1500:
            max_end = 200
        else:
            max_end = 400

    toc_start = None

    for i, line in enumerate(lines):
        cleaned = line.strip()
        if cleaned and COMPILED_PATTERNS['toc_head_pattern'].match(cleaned):
            toc_start = i
            break

    if toc_start is None:
        for i, line in enumerate(lines[10:max_start]):
            if is_probably_toc_start(lines, start=i+10):
                toc_start = i+10
                break

    if toc_start is None:
        return text, []

    toc_end = None
    for j in range(toc_start + 1, min(toc_start + max_end, len(lines))):
        line = lines[j].strip()
        if not line:
            continue

        line = re.sub(r'^\s*(\d+[\.\)]?)+(\s+|$)', '', line)
        if len(line.split()) >= 10 and line.count('.') <= 2:
            toc_end = j-2
            break

    if toc_end is None:
        toc_end = min(toc_start + max_end, len(lines))

    toc = '\n'.join(lines[toc_start:toc_end])
    if len(toc) < 100:
        return text, []

    cleaned_lines = lines[:toc_start] + lines[toc_end:]

    return '\n'.join(cleaned_lines).strip(), [toc]


def is_probably_lit_line(lines, start, window=3, thresh=0.2, debug=False):
    """
    Determines whether a window of lines likely belongs to a literature/reference list.
    Checks for capitalization ratio, links, numeric content, and body-like structure.
    """
    slice_lines = lines[start:start + window]
    if not slice_lines:
        return 0

    joined = ' '.join(slice_lines)
    words = joined.split()
    if not words:
        return 0
    cap_frac = sum(1 for word in words if word[0].isupper()) / len(words)
    links = len(COMPILED_PATTERNS['link_pattern'].findall(joined))
    body_like = any(len(line.split()) >= 6 for line in slice_lines)
    digits = any(sum(c.isdigit() for c in line) > 3 for line in slice_lines)
    if debug:
        print(f"[line {start}] cap_frac: {cap_frac:.2f}, links: {links}, body_like: {body_like}, digits>3: {digits}")
    return (cap_frac > thresh) or (links > 0) or (not body_like) or digits


def clean_literature_appendix(text):
    """
    Removes literature or bibliography sections from the text based on keyword matches and structure heuristics.
    Returns cleaned text and a list of removed sections.
    """
    if len(text) < 500:
        return text, []

    lines = text.split('\n')
    line_count = len(lines)

    skip = int(line_count * 0.2)
    cleaned_lines = lines[:skip]
    removed_sections = []

    i = skip
    while i < line_count:
        line = lines[i]
        cleaned = line.replace('\x0c', '').strip()

        if COMPILED_PATTERNS['literature_pattern'].match(cleaned):
            if i > 2000:
                removed_sections.append('\n'.join(lines[i:]).strip())
                return '\n'.join(cleaned_lines).strip(), removed_sections

            start = i
            end = i + 1
            while end < line_count:
                next_line = lines[end]
                cleaned_next = next_line.replace('\x0c', '').strip()

                if (COMPILED_PATTERNS['bilag_pattern'].match(cleaned_next)
                    and i > line_count * 0.5):
                    removed_sections.append('\n'.join(lines[start:]).strip())
                    return '\n'.join(cleaned_lines).strip(), removed_sections

                if is_probably_lit_line(lines, end, window=min(3, line_count - end)):
                    end += 1
                else:
                    break

            removed = '\n'.join(lines[start:end]).strip()
            if removed:
                removed_sections.append(removed)

            i = end
        else:
            cleaned_lines.append(line)
            i += 1

    return '\n'.join(cleaned_lines).strip(), removed_sections


def clean_appendix(text):
    """
    Removes appendix sections from the second half of the document based on keyword matches.
    Returns cleaned text and removed sections.
    """
    lines = text.split('\n')

    if len(text) < 1000 or len(lines) < 150:
        return text, []

    min_index = int(len(lines) * 0.5)

    for i in range(min_index, len(lines)):
        line = lines[i]
        cleaned = line.replace('\x0c', '').strip()

        if COMPILED_PATTERNS['bilag_pattern'].match(cleaned):
            removed = '\n'.join(lines[i:]).strip()
            kept = '\n'.join(lines[:i]).strip()
            return kept, [removed]

    return text, []


def remove_repeated_sentences(text):
    """
    Removes duplicate sentences from the text while preserving the first occurrence of each.
    """
    sentences = sent_tokenize(text, language="english")
    seen = set()
    unique_sentences = []

    for sentence in sentences:
        cleaned = sentence.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique_sentences.append(cleaned)

    return ' '.join(unique_sentences)


def remove_page_footnotes(text, bottom_n=10):
    """
    Removes likely footnotes from the bottom `bottom_n` lines of each page.
    Matches patterns typical of citations or references.
    """
    pages = text.split('\x0c')
    cleaned_pages = []

    for page in pages:
        lines = page.split('\n')
        line_count = len(lines)

        if line_count <= bottom_n:
            cleaned_pages.append(page.strip())
            continue

        top_lines = lines[:-bottom_n]
        bottom_lines = lines[-bottom_n:]

        cleaned_bottom = [
            line for line in bottom_lines
            if not COMPILED_PATTERNS['footnote_pattern'].match(line.strip())
        ]

        cleaned_page = '\n'.join(top_lines + cleaned_bottom).strip()
        cleaned_pages.append(cleaned_page)

    return '\x0c'.join(cleaned_pages).strip()


def remove_fuzzy_duplicated_page_lines(text, top_n=7, bottom_n=7, min_repeats=2, similarity_threshold=90):
    """
    Removes repeated header/footer lines from pages using fuzzy matching.
    Groups similar lines and deletes those appearing on at least `min_repeats` pages.
    """
    pages = text.split('\x0c')
    candidate_lines = []

    for page in pages:
        lines = page.split('\n')
        header = lines[:top_n]
        footer = lines[-bottom_n:] if bottom_n > 0 else []
        for line in header + footer:
            cleaned = line.strip()
            if cleaned:
                candidate_lines.append(cleaned)

    groups = defaultdict(list)

    for line in candidate_lines:
        matched = False
        for rep in groups:
            if fuzz.ratio(line, rep) >= similarity_threshold:
                groups[rep].append(line)
                matched = True
                break
        if not matched:
            groups[line].append(line)

    line_page_map = defaultdict(set)

    for page_idx, page in enumerate(pages):
        lines = set(l.strip() for l in page.split('\n')[:top_n] + page.split('\n')[-bottom_n:] if l.strip())
        for group_rep, variants in groups.items():
            for var in variants:
                if any(fuzz.ratio(var, line) >= similarity_threshold for line in lines):
                    line_page_map[group_rep].add(page_idx)

    repeated_groups = {rep for rep, pages_set in line_page_map.items() if len(pages_set) >= min_repeats}

    cleaned_pages = []
    for page in pages:
        lines = page.split('\n')
        cleaned_lines = []
        for line in lines:
            cleaned = line.strip()
            if cleaned and any(fuzz.ratio(cleaned, rep) >= similarity_threshold for rep in repeated_groups):
                continue
            cleaned_lines.append(line)
        cleaned_pages.append('\n'.join(cleaned_lines).strip())

    return '\x0c'.join(cleaned_pages).strip()


def remove_page_number_lines(text, bottom_n=10):
    """
    Removes page number lines from the last `bottom_n` lines of each page.
    """

    pages = text.split('\x0c')
    cleaned_pages = []

    for page in pages:
        lines = page.split('\n')
        line_count = len(lines)

        if line_count <= bottom_n:
            cleaned_pages.append(page.strip())
            continue

        top_lines = lines[:-bottom_n]
        bottom_lines = lines[-bottom_n:]

        cleaned_bottom = [
            line for line in bottom_lines
            if not COMPILED_PATTERNS['page_number_pattern'].match(line.strip())
        ]

        cleaned_pages.append('\n'.join(top_lines + cleaned_bottom).strip())

    return '\n\x0c'.join(cleaned_pages).strip()


def remove_in_text_citations(text):
    """
    Removes in-text citations including:
    - Numeric citations like [1], [2,3]
    - Parentheses with a 4-digit year or 'ibid'
    - Cleans up leftover punctuation and spacing
    """
    text = COMPILED_PATTERNS['numeric_citations'].sub('', text)
    text = COMPILED_PATTERNS['year_citations'].sub('', text)
    text = COMPILED_PATTERNS['empty_parens'].sub('', text)

    text = COMPILED_PATTERNS['multiple_spaces'].sub(' ', text)
    text = COMPILED_PATTERNS['space_before_punct'].sub(r'\1', text)
    text = COMPILED_PATTERNS['double_punct'].sub(r'\1', text)
    text = COMPILED_PATTERNS['trailing_space'].sub('', text)

    return text.strip()


def remove_leading_numbers(text):
    """Remove leading numbers and optional whitespace from the beginning of a string."""
    return COMPILED_PATTERNS['leading_numbers'].sub('', text)


def finalize_text(text):
    """
    Performs final cleanup of text: normalizes quotes, removes unwanted symbols, replaces URLs/emails/phone numbers,
    fixes spacing issues, and strips extra whitespace.
    """

    replacements = [
    (r"\\(['\"])", r"\1"),
    ("’", "'"), ("‘", "'"), ('“', '"'), ('”', '"'),
    ('´', "'"), ('`', "'"), ('«', '"'), ('»', '"'),
    ('\uf0b7', ''), ('ﬁ', 'fi'), ('ﬂ', 'fl')]


    for old, new in replacements:
        try:
            text = re.sub(old, new, text)
        except re.error:
            text = text.replace(old, new)

    text = re.sub(r"\\(['\"])(.*?)\\\1", r"\g<1>\g<2>\g<1>", text)
    text = re.sub(r"\\(['\"])", r"\1", text)
    text = re.sub(r'\\(?!["\'])', '', text)

    text = COMPILED_PATTERNS['url_replace'].sub('URL', text)
    text = COMPILED_PATTERNS['email_replace'].sub('EMAILADRESS', text)
    text = COMPILED_PATTERNS['phone_replace'].sub('PHONENUMBER', text)
    text = COMPILED_PATTERNS['long_dashes'].sub(' ', text)
    text = COMPILED_PATTERNS['hyphen_break'].sub('', text)
    text = COMPILED_PATTERNS['tab_spaces_final'].sub(' ', text)
    text = COMPILED_PATTERNS['multiple_spaces'].sub(' ', text)

    return text.strip()


# === Sentence extraction & filtering ===

def extract_sentences_from_text(text):
    paragraphs = text.split('\n\n')
    all_sentences = []

    for paragraph in paragraphs:
        flat_paragraph = paragraph.replace('\n', ' ').strip()
        flat_paragraph = flat_paragraph.replace('\x0c', ' ').strip()
        # Apply English abbreviation cleaning BEFORE tokenizing
        flat_paragraph = remove_english_abbrev_dots(flat_paragraph)

        sentences = sent_tokenize(flat_paragraph, language='english')
        all_sentences.extend(sentences)

    return all_sentences


def clean_sentences(sentences):
    cleaned = []
    for doc in nlp.pipe(sentences, batch_size=1000):
        if any(t.pos_ == "VERB" for t in doc):
            cleaned.append(doc.text)
    return " ".join(cleaned)


# === Cleaning pipeline ===

steps = ['fmt', 'toc', 'lit', 'app']

c_parts = {}
c_stats = {}

def clean_string(text: str, bid: str) -> str:
    c_stats[bid] = {'original': len(text)}
    c_parts[bid] = {step: '' for step in steps}

    #print(f"\n=== Cleaning BID {bid} ===")
    #print(f"[original] length = {len(text)}")

    text, removed_fmt = fmt_check(text)
    c_parts[bid]['fmt'] = removed_fmt
    c_stats[bid]['after_fmt'] = len(text)
    #print(f"[after_fmt] length = {len(text)}")

    text = split_at_intro(text)
    c_stats[bid]['after_intro'] = len(text)
    #print(f"[after_intro] length = {len(text)}")

    text, removed_toc = clean_toc(text)
    c_parts[bid]['toc'] = removed_toc
    c_stats[bid]['after_toc'] = len(text)
    #print(f"[after_toc] length = {len(text)}")

    text = remove_fuzzy_duplicated_page_lines(text)
    c_stats[bid]['after_dup_page'] = len(text)
    #print(f"[after_dup_page] length = {len(text)}")

    text = remove_page_footnotes(text)
    c_stats[bid]['after_footnotes'] = len(text)
    #print(f"[after_footnotes] length = {len(text)}")

    text = remove_page_number_lines(text)
    c_stats[bid]['after_pg_num'] = len(text)
    #print(f"[after_pg_num] length = {len(text)}")

    text, removed_lit = clean_literature_appendix(text)
    c_parts[bid]['lit'] = removed_lit
    c_stats[bid]['after_lit'] = len(text)
    #print(f"[after_lit] length = {len(text)}")

    text, removed_app = clean_appendix(text)
    c_parts[bid]['app'] = removed_app
    c_stats[bid]['after_app'] = len(text)
    #print(f"[after_app] length = {len(text)}")

    text = remove_in_text_citations(text)
    c_stats[bid]['after_in_text_citations'] = len(text)
    #print(f"[after_in_text_citations] length = {len(text)}")

    text = remove_leading_numbers(text)
    c_stats[bid]['after_leading_numbers'] = len(text)
    #print(f"[after_leading_numbers] length = {len(text)}")

    text = finalize_text(text)
    c_stats[bid]['after_finalize'] = len(text)
    #print(f"[after_finalize] length = {len(text)}")

    text = remove_repeated_sentences(text)
    c_stats[bid]['after_repeated_sent'] = len(text)
    #print(f"[after_repeated_sent] length = {len(text)}")

    return text




# === Streamed cleaning ===

def clean_texts_streamed(input_file, output_file,
                         cparts_file="c_parts.json",
                         cstats_file="c_stats.parquet",
                         key_field='bid', value_field='text'):
    """
    Streams a JSONL file line-by-line, cleans the text, writes cleaned output,
    and saves c_parts as JSON and c_stats as parquet (df).
    """

    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:

        for line in tqdm(fin, desc="Cleaning and writing", total=total_lines):
            obj = json.loads(line)
            key = obj[key_field]
            raw_text = obj[value_field]

            cs = clean_string(raw_text, key)
            sentences = extract_sentences_from_text(cs)
            cleaned = clean_sentences(sentences)

            json_line = json.dumps({key_field: key, value_field: cleaned})
            fout.write(json_line + '\n')
            
    with open(cparts_file, "w", encoding="utf-8") as f:
        json.dump(c_parts, f, ensure_ascii=False, indent=2)

    df_cleaned = pd.DataFrame.from_dict(c_stats, orient="index")
    ordered_steps = ['original'] + [col for col in df_cleaned.columns if col.startswith('after_') and col in df_cleaned.columns]

    for prev, curr in zip(ordered_steps, ordered_steps[1:]):
        step_name = curr.replace('after_', '')
        delta_col = f"delta_{step_name}"
        pct_col = f"pct_{step_name}"
        df_cleaned[delta_col] = df_cleaned[prev] - df_cleaned[curr]
        df_cleaned[pct_col] = df_cleaned.apply(
            lambda row: round(row[delta_col] / row['original'], 3) if row['original'] > 0 else 0,
            axis=1
        )

    pct_cols = [c for c in df_cleaned.columns if c.startswith('pct_')]
    df_cleaned['total_reduction'] = df_cleaned[pct_cols].sum(axis=1)

    df_cleaned.reset_index().rename(columns={"index": "bid"}).to_parquet(cstats_file, index=False)

    print(f"✅ Updated cleaned output → {output_file}")
    print(f"✅ Saved c_parts → {cparts_file}")
    print(f"✅ Saved c_stats → {cstats_file}")


def resume_clean_texts(input_file, output_file,
                       cparts_file="c_parts.json",
                       cstats_file="c_stats.parquet",
                       key_field='bid', value_field='text'):
    """
    Resumes cleaning, skipping already processed entries.
    Also saves c_parts and c_stats at the end.
    """
    existing_bids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as fout:
            for line in fout:
                try:
                    obj = json.loads(line)
                    existing_bids.add(obj[key_field])
                except Exception:
                    continue
        print(f"✅ Found {len(existing_bids):,} entries already written. Skipping those.")

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'a', encoding='utf-8') as fout:

        for line in tqdm(fin, desc="Resuming cleaning"):
            try:
                obj = json.loads(line)
                bid = obj[key_field]
                raw_text = obj[value_field]
            except Exception:
                continue

            if bid in existing_bids:
                continue

            cs = clean_string(raw_text, bid)
            sentences = extract_sentences_from_text(cs)
            cleaned = clean_sentences(sentences)

            json_line = json.dumps({key_field: bid, value_field: cleaned})
            fout.write(json_line + '\n')

    with open(cparts_file, "w", encoding="utf-8") as f:
        json.dump(c_parts, f, ensure_ascii=False, indent=2)

    df_cleaned = pd.DataFrame.from_dict(c_stats, orient="index")
    ordered_steps = ['original'] + [col for col in df_cleaned.columns if col.startswith('after_') and col in df_cleaned.columns]

    for prev, curr in zip(ordered_steps, ordered_steps[1:]):
        step_name = curr.replace('after_', '')
        delta_col = f"delta_{step_name}"
        pct_col = f"pct_{step_name}"
        df_cleaned[delta_col] = df_cleaned[prev] - df_cleaned[curr]
        df_cleaned[pct_col] = df_cleaned.apply(
            lambda row: round(row[delta_col] / row['original'], 3) if row['original'] > 0 else 0,
            axis=1
        )

    pct_cols = [c for c in df_cleaned.columns if c.startswith('pct_')]
    df_cleaned['total_reduction'] = df_cleaned[pct_cols].sum(axis=1)

    df_cleaned.reset_index().rename(columns={"index": "bid"}).to_parquet(cstats_file, index=False)

    print(f"✅ Updated cleaned output → {output_file}")
    print(f"✅ Saved c_parts → {cparts_file}")
    print(f"✅ Saved c_stats → {cstats_file}")


if args.resume:
    resume_clean_texts(
        input_file=input_path,
        output_file=output_path,
        cstats_file=cstats_file,
        cparts_file=cparts_file,
        key_field='bid',
        value_field='text'
    )
else:
    clean_texts_streamed(
        input_file=input_path,
        output_file=output_path,
        cstats_file=cstats_file,
        cparts_file=cparts_file,
        key_field='bid',
        value_field='text'
    )
