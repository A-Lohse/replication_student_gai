#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install fastparquet


# In[2]:


import os, json
import pandas as pd
from pathlib import Path
import numpy as np
import re
import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
from collections import Counter
from zoneinfo import ZoneInfo
from ast import literal_eval
import ast
from tqdm import tqdm
# =====================
# Paths (adjust if needed)
# =====================
BASE = "/work/data_assigments/DATA"
DATA_PATH = Path("/work/data_assigments/DATA/")

ALL_DATA_CSV        = os.path.join(BASE, "data_all.csv")
BID_META_CSV        = os.path.join(BASE, "bid_meta_df.csv")
EN_CLEAN_JSONL      = os.path.join(BASE, "all_english_entries_clean.jsonl")  # {bid, text}

TZ = ZoneInfo("Europe/Copenhagen")

# In[3]:


def load_english_bids(jsonl_path: str) -> list:
    """Return unique BIDs from a JSONL file by reading it line by line."""
    bids = set()
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            bids.add(str(data["bid"]))
    return list(bids)


# In[4]:


# =====================
# 1) Load + filter base metadata (KU/RUC) and intersect with bid_meta
# =====================
print("🔹 Loading metadata...")
all_data = pd.read_csv(ALL_DATA_CSV, dtype={"bid": str})
all_data = all_data.loc[all_data["university"].isin(["KU", "RUC"])].copy()
print(f"  all_data (KU/RUC): {all_data.shape}")


# In[5]:


bid_meta = pd.read_csv(BID_META_CSV, dtype={"bid": str}).drop_duplicates("bid")
bid_meta = bid_meta.loc[bid_meta["len"] > 0].copy()
print(f"  bid_meta (len>0): {bid_meta.shape}")

final_meta = all_data.loc[all_data["bid"].isin(bid_meta["bid"])].copy()
print(f"  final_meta (intersection): {final_meta.shape}")


# In[6]:


# =====================
# 2) Keep only rows that actually have English text available
# =====================
print("🔹 Reading English BIDs from cleaned JSONL...")
eng_bids = load_english_bids(EN_CLEAN_JSONL)


# In[7]:


final_meta["has_text"] = final_meta["bid"].astype(str).isin(eng_bids)


# In[8]:


final_meta["has_text"] = final_meta["has_text"]*1


# In[9]:


print(f"  has_text=1 count: {final_meta['has_text'].sum()} / {len(final_meta)}")


# In[10]:


final_meta = final_meta.loc[final_meta["has_text"] == 1].copy()
final_meta["AfleveringsTidspunkt"] = pd.to_datetime(final_meta["AfleveringsTidspunkt"], errors="coerce")
final_meta["AfleveringsTidspunkt"] = final_meta["AfleveringsTidspunkt"].dt.tz_localize(TZ) 
# In[11]:


# Optional: keep some IDs as strings (mirrors your DK flow)
for col in ("Kursuskode"):
    if col in all_data.columns:
        final_meta[col] = all_data[col].astype(str)


# In[12]:


# Rename columns for consistency
final_meta = final_meta.rename({"Kursuskode": "examncode", "Kursusnavn": "examnname"}, axis=1)

# Load mapping from BID to course ID
file_path = DATA_PATH / "besvarelsesid2courseid.js"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()
    bid_to_course = json.loads(content)
    bid2courseid = {str(k): v for k, v in bid_to_course.items() if v}  # Remove empty values

# Map course code onto dataset from bid-to-course mapping
final_meta["coursecode"] = final_meta["bid"].map(bid2courseid)
final_meta["coursecode"] = final_meta["coursecode"].fillna(final_meta["examncode"])
final_meta["coursecode_noE"] = final_meta["coursecode"].str.rstrip("E")
final_meta["coursecode_clean"] = (final_meta["coursecode"]
    .str.replace(r'(A|GB|\.0)$', '', regex=True)
    .str.strip()      # remove leading/trailing whitespace
    .str.upper()      # normalize case
)


handin_data = final_meta.copy()

print(f"🧩 Added coursecode to dataset. Missing coursecodes: {final_meta.coursecode.isna().sum()}")
print(f"🧩 Added examncodes to dataset. Missing examncodes: {final_meta.examncode.isna().sum()}")


# In[13]:
ku_course_data = pd.read_pickle(DATA_PATH / "ku_coursedata_eid3_is_courseid.pkl")
ruc_course_data = pd.read_pickle(DATA_PATH / "ruc_coursedata_eid3_is_courseid.pkl")

ku_course_data.reset_index(inplace = True)
ruc_course_data.reset_index(inplace = True)


ku_course_data.replace("", np.nan, inplace=True)
ruc_course_data.replace("", np.nan, inplace=True)

ruc_course_data = ruc_course_data.loc[~ruc_course_data["eid3"].isna()]
ku_course_data = ku_course_data.loc[~ku_course_data["eid3"].isna()]



ruc_course_data["course_title_text"] = ruc_course_data["name"].apply(
    lambda d: d.get("Kursus titel") if isinstance(d, dict) else None
)

ku_course_data["course_title_text"] = ku_course_data["name"].apply(
    lambda d: d.get("course_name") if isinstance(d, dict) else None
)


coursedata = pd.concat([ku_course_data[["name","exam", "year", "semester", "eid", "eid2", "eid3"]],
                        ruc_course_data[["name", "exam", "year", "semester", "eid", "eid2", "eid3"]]])

coursedata = coursedata.replace({"" : None})

coursedata["courseid"] = coursedata["eid3"]
coursedata["courseid"] = coursedata["courseid"].fillna(coursedata["eid2"])
coursedata["courseid"] = coursedata["courseid"].fillna(coursedata["eid"])

coursedata["coursecode_clean"] = (
    coursedata["courseid"]
    .str.replace(r'(A|GB|\.0)$', '', regex=True)
    .str.strip()      # remove leading/trailing whitespace
    .str.upper()      # normalize case
)

coursedata = coursedata.dropna(subset=["coursecode_clean"]).copy()

# Add a running number per coursecode_clean
coursedata["exam_code"] = (
    coursedata.groupby("coursecode_clean").cumcount()
)

# Combine with coursecode_clean
coursedata["exam_code"] = (
    coursedata["coursecode_clean"] + "_" + coursedata["exam_code"].astype(str)
)

relevant_coursedata = coursedata.loc[coursedata["coursecode_clean"].isin(handin_data["coursecode_clean"].unique())]
relevant_coursedata.reset_index(inplace=True)

print("🏛️ Loaded KU and RUC course metadata")


print("🏛️ Mapping handins to exams (not simply courses)")

relevant_coursedata['year_clean'] = relevant_coursedata['year']

year_counts = (
    relevant_coursedata.groupby('coursecode_clean', as_index=False)['year_clean']
                       .nunique()
                       .rename(columns={'year_clean': 'n_years'})
)
multi_codes = year_counts.loc[year_counts['n_years'] > 1, 'coursecode_clean']
cm = relevant_coursedata[relevant_coursedata['coursecode_clean'].isin(multi_codes)].copy()

# ---------------------------------------
# 1) From `semester`: F/E/Forår/Efterår
# ---------------------------------------
def _semester_text(x):
    if isinstance(x, dict):
        return x.get('Semester') or x.get('semester')
    if isinstance(x, str):
        try:
            d = ast.literal_eval(x)
            if isinstance(d, dict):
                return d.get('Semester') or d.get('semester')
            return x
        except Exception:
            return x
    return None

cm['sem_str'] = cm['semester'].apply(_semester_text)

is_Fcode  = cm['sem_str'].astype(str).str.fullmatch(r'[Ff]\d{4}', na=False)
is_Ecode  = cm['sem_str'].astype(str).str.fullmatch(r'[Ee]\d{4}', na=False)
is_foraar = cm['sem_str'].str.contains(r'forår|spring', case=False, na=False)
is_efter  = cm['sem_str'].str.contains(r'efterår|autumn|fall', case=False, na=False)

cm['block_from_sem'] = np.select(
    [is_Fcode | is_foraar, is_Ecode | is_efter],
    [3, 1],
    default=np.nan
)

# -------------------------------------------------
# 2) From Placement/Placering: Block 1..4 (+ multi)
# -------------------------------------------------
def _placement_text(x):
    if isinstance(x, dict):
        return x.get('Placement') or x.get('Placering')
    if isinstance(x, str):
        try:
            d = ast.literal_eval(x)
            if isinstance(d, dict):
                return d.get('Placement') or d.get('Placering')
            return x
        except Exception:
            return x
    return None

cm['place_str'] = cm['semester'].apply(_placement_text)

def _extract_blocks(s):
    if not isinstance(s, str):
        return []
    # "Block 1, Block 2", "Blok 3", "1", etc.
    nums = re.findall(r'(?i)(?:blok|block)?\s*\b([1-4])\b', s)
    return sorted({int(n) for n in nums})

def _term_from_blocks(blocks):
    if not blocks:
        return None
    has_fall = any(b in (1, 2) for b in blocks)
    has_spring = any(b in (3, 4) for b in blocks)
    if has_fall and has_spring:
        return 'full_year'
    if has_fall:
        return 'fall'
    if has_spring:
        return 'spring'
    return None

cm['blocks_list'] = cm['place_str'].apply(_extract_blocks)
cm['term_from_place'] = cm['blocks_list'].apply(_term_from_blocks)
cm['block_from_place'] = cm['term_from_place'].map({'fall': 1, 'spring': 3, 'full_year': 1})

# Preferred: semester → placement
cm['block_guess'] = cm['block_from_sem'].combine_first(cm['block_from_place'])

# ---------------------------------------------
# 3) Fill remaining via free text (weeks/months)
# ---------------------------------------------
missing_mask = cm['block_guess'].isna()

# Normalize text
cm.loc[missing_mask, 'place_norm'] = (
    cm.loc[missing_mask, 'place_str'].fillna('')
      .str.replace(r'\s+', ' ', regex=True)
      .str.strip()
)

# Season words (EN+DA)
has_autumn_w = cm.loc[missing_mask, 'place_norm'].str.contains(r'\b(?:autumn|efterår|fall)\b', case=False, na=False)
has_spring_w = cm.loc[missing_mask, 'place_norm'].str.contains(r'\b(?:spring|forår)\b', case=False, na=False)
has_summer_w = cm.loc[missing_mask, 'place_norm'].str.contains(r'\b(?:summer|sommer)\b', case=False, na=False)
has_year_w   = cm.loc[missing_mask, 'place_norm'].str.contains(r'\b(?:hele året|through ?out the year|whole year|all year)\b', case=False, na=False)

# Months (EN+DA)
has_spring_m = cm.loc[missing_mask, 'place_norm'].str.contains(r'\b(?:mar(?:ch)?|marts|apr(?:il)?|maj|may|jun(?:e)?|juni)\b', regex=True)
has_summer_m = cm.loc[missing_mask, 'place_norm'].str.contains(r'\b(?:jul(?:y)?|juli|aug(?:ust)?|august)\b', regex=True)
has_autumn_m = cm.loc[missing_mask, 'place_norm'].str.contains(r'\b(?:sep(?:t|tember)?|okt(?:ober)?|oct(?:ober)?|nov(?:ember)?)\b', regex=True)
has_winter_m = cm.loc[missing_mask, 'place_norm'].str.contains(r'\b(?:dec(?:ember)?|jan(?:uary)?|januar|feb(?:ruary)?|februar)\b', regex=True)

# Week numbers → coarse season
def _extract_weeks(s):
    if not isinstance(s, str):
        return []
    nums = re.findall(r'(?i)\b(?:uge|week)\s*(\d{1,2})\b', s)
    out = []
    for n in nums:
        try:
            v = int(n)
            if 1 <= v <= 53:
                out.append(v)
        except Exception:
            pass
    return out

weeks_list = cm.loc[missing_mask, 'place_norm'].apply(_extract_weeks)

def _weeks_to_term(weeks):
    if not weeks:
        return None
    has_aut = any((w >= 36 and w <= 52) or (w >= 1 and w <= 5) for w in weeks)
    has_spr = any(6 <= w <= 25 for w in weeks)
    has_sum = any(26 <= w <= 35 for w in weeks)
    flags = {'fall': has_aut, 'spring': has_spr, 'summer': has_sum}
    on = [k for k, v in flags.items() if v]
    if len(on) == 0:
        return None
    if len(on) > 1:
        return 'full_year'
    return on[0]

term_from_weeks = weeks_list.apply(_weeks_to_term)

# Decide term per row (priority: whole-year phrase, then weeks, then words/months)
def _decide_term(idx):
    if has_year_w.loc[idx]:
        return 'full_year'
    tw = term_from_weeks.loc[idx]
    if isinstance(tw, str):
        return tw
    pos = set()
    if has_autumn_w.loc[idx] or has_autumn_m.loc[idx]:
        pos.add('fall')
    if has_spring_w.loc[idx] or has_spring_m.loc[idx]:
        pos.add('spring')
    if has_summer_w.loc[idx] or has_summer_m.loc[idx]:
        pos.add('summer')
    # winter months only (often Jan exams) → fall unless other signals exist
    if has_winter_m.loc[idx] and not pos:
        pos.add('fall')
    if len(pos) == 0:
        return None
    if len(pos) > 1:
        return 'full_year'
    return next(iter(pos))

idxs = cm.index[missing_mask]
term_from_text = pd.Series({i: _decide_term(i) for i in idxs})

fill_series = term_from_text.map({'fall': 1, 'spring': 3, 'summer': 5, 'full_year': 1})
cm['block_guess'] = cm['block_guess']
cm['block_guess'].update(fill_series)

# ---------------------------------------------
# 4) Final pass: Danish "foråret", month ranges
# ---------------------------------------------
rem = cm['block_guess'].isna()
if rem.any():
    s = cm.loc[rem, 'place_norm'].fillna('')

    # quick win: explicit Danish "foråret" (in spring)
    has_foraaret = s.str.contains(r'\bforåret\b', case=False, na=False)

    # month mapping (EN + DA)
    month_map = {
        'january':1,'jan':1,'januar':1,
        'february':2,'feb':2,'februar':2,
        'march':3,'mar':3,'marts':3,
        'april':4,'apr':4,
        'may':5,'maj':5,
        'june':6,'jun':6,'juni':6,
        'july':7,'jul':7,'juli':7,
        'august':8,'aug':8,'august':8,
        'september':9,'sep':9,'sept':9,
        'october':10,'oct':10,'oktober':10,'okt':10,
        'november':11,'nov':11,
        'december':12,'dec':12,'december':12
    }

    # regex to find months and optional day numbers/ordinals
    # examples matched: "31 August", "25 September 2020", "30 July to 10 August", "19-23 November 2018", "August 8th to 12th + 23th"
    month_token = r'(january|jan(?:uary)?|januar|february|feb(?:ruary)?|februar|march|mar(?:ch)?|marts|april|apr(?:il)?|may|maj|june|jun(?:e)?|juni|july|jul(?:y)?|juli|august|aug(?:ust)?|september|sep(?:t|tember)?|october|oct(?:ober)?|okt(?:ober)?|november|nov|december|dec)'
    day_token   = r'(\d{1,2})(?:st|nd|rd|th)?'
    # capture either "DAY MONTH" or "MONTH DAY" or just "MONTH"
    pat_months = re.compile(
        rf'\b(?:{day_token}\s+{month_token}|{month_token}\s+{day_token}|{month_token})\b',
        flags=re.IGNORECASE
    )

    def months_in_text(txt: str):
        txt = str(txt)
        found = []
        for m in pat_months.finditer(txt):
            # the month can be in group 2 or 1 or 3 depending on which alternative matched
            groups = m.groups()
            # groups order: (day?, month?, month?, day?) due to alternation; so collect all month-like groups
            for g in groups:
                if not g:
                    continue
                key = g.lower()
                if key in month_map:
                    found.append(month_map[key])
        # de-duplicate
        return sorted(set(found))

    months_list = s.apply(months_in_text)

    def months_to_term(months):
        if not months:
            return None
        # map months to coarse seasons
        has_spring = any(m in (3,4,5,6) for m in months)   # Mar–Jun
        has_summer = any(m in (7,8) for m in months)        # Jul–Aug
        has_autumn = any(m in (9,10,11) for m in months)    # Sep–Nov
        has_winter = any(m in (12,1,2) for m in months)     # Dec–Feb

        flags = {'fall': has_autumn or (has_winter and not (has_spring or has_summer)),
                 'spring': has_spring,
                 'summer': has_summer}
        on = [k for k,v in flags.items() if v]
        if len(on) == 0:
            return None
        if len(on) > 1:
            return 'full_year'
        return on[0]

    term_from_months = months_list.apply(months_to_term)

    # combine: foråret wins → spring; else month-derived term
    final_term = pd.Series(index=s.index, dtype=object)
    final_term.loc[has_foraaret[has_foraaret].index] = 'spring'
    # fill remaining from months
    need = final_term.isna()
    final_term.loc[need] = term_from_months.loc[need]

    # map to blocks (your rule: full_year -> 1)
    final_blocks = final_term.map({'fall':1, 'spring':3, 'summer':5, 'full_year':1}).astype('Float64')

    # update block_guess (index-aligned, safe)
    cm['block_guess'] = cm['block_guess'].astype('Float64')
    cm['block_guess'].update(final_blocks)


# merge back into relevant_coursedata
relevant_coursedata = relevant_coursedata.merge(
    cm[['exam_code', 'block_guess']],  # keep exam_code as key
    on='exam_code',
    how='left',
    suffixes=('', '_from_multi')
)

# if we got a guess, use it, else fall back to 1
relevant_coursedata['block_final'] = (
    relevant_coursedata['block_guess']
        .fillna(1) 
)

# ------------------------------------------
# 0) Build a course_start_dt on course entries
# ------------------------------------------
# ensure block_final exists (and default 1 already applied earlier)
relevant_coursedata = relevant_coursedata.dropna(subset=['year', 'block_final']).copy()
relevant_coursedata['year'] = relevant_coursedata['year'].astype('Int64')
relevant_coursedata['block_final'] = relevant_coursedata['block_final'].astype('Int64')

# deterministically collapse overlaps within same (coursecode_clean, year, block_final)
# (keep the first in the current order; change sort if you want another policy) - chosen as they where all similar under manual inspection
relevant_coursedata = (
    relevant_coursedata.sort_values(['coursecode_clean', 'year', 'block_final', 'exam_code'])
           .drop_duplicates(['coursecode_clean', 'year', 'block_final'], keep='first')
           .reset_index(drop=True)
)


# rule: Block 1 (fall) -> Jan 31 (next year); Block 3 (spring) -> Jun 30 (same year)
relevant_coursedata['course_start_dt'] = pd.NaT

# Block 1 = Fall (start Sep 1 of given year)
mask_b1 = relevant_coursedata['block_final'] == 1
relevant_coursedata.loc[mask_b1, 'course_start_dt'] = pd.to_datetime({
    'year': relevant_coursedata.loc[mask_b1, 'year'].astype(int),
    'month': 9, 'day': 1
})

# Block 3 = Spring (start Feb 1 of given year)
mask_b3 = relevant_coursedata['block_final'] == 3
relevant_coursedata.loc[mask_b3, 'course_start_dt'] = pd.to_datetime({
    'year': relevant_coursedata.loc[mask_b3, 'year'].astype(int),
    'month': 2, 'day': 1
})

# (optionally summer/full_year rules if you have them)


# localize to Copenhagen
relevant_coursedata['course_start_dt'] = relevant_coursedata['course_start_dt'].dt.tz_localize(TZ)

# sort for merge
relevant_coursedata = relevant_coursedata.sort_values(['coursecode_clean', 'course_start_dt']).reset_index(drop=True)


handin_data = handin_data.sort_values(['coursecode_clean', 'AfleveringsTidspunkt']).reset_index(drop=True)


print("🏛️ Mapping each hand in to first prior course description")
mapped_rows = []

for i, row in tqdm(handin_data.iterrows()):
    code = row['coursecode_clean']
    t = row['AfleveringsTidspunkt']

    cdf = relevant_coursedata.loc[relevant_coursedata['coursecode_clean'] == code]

    # only keep courses that start before hand-in
    cdf = cdf[cdf['course_start_dt'] <= t]

    if cdf.empty:
        mapped_rows.append(pd.NA)
    else:
        # pick the one with the latest start before t
        match = cdf.sort_values('course_start_dt').iloc[-1]
        mapped_rows.append(match['exam_code'])

handin_data['mapped_exam_code'] = mapped_rows
handin_data = handin_data.dropna(subset = ['mapped_exam_code'])

bid2examcode = dict(zip(handin_data["bid"],handin_data["mapped_exam_code"]))

print("📊 Missing values in course dataset:")
print(relevant_coursedata.isna().sum())
print(relevant_coursedata.shape)

print("💾 Saved course dataset to 'english_coursedata.parquet'")
course_dataset_with_data = relevant_coursedata.dropna()
course_dataset_with_data.to_parquet(DATA_PATH / "english_coursedata.parquet")

print("📊 Missing values in course dataset (dropped NA):")
print(course_dataset_with_data.isna().sum())
print(course_dataset_with_data.shape)


# Export

print("💾 Saved bid2examcode mapping to 'bid2examcode.json'")
outpath = Path(DATA_PATH) / "bid2examcode.json"
with open(outpath, "w", encoding="utf-8") as f:
    json.dump(bid2examcode, f, ensure_ascii=False, indent=2)


print("\n🎉 Script completed successfully!")




