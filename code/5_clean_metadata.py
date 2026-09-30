#!/usr/bin/env python
# coding: utf-8

# In[6]:


#!pip install fastparquet


# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:




# In[1]:


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


# In[2]:


# =====================
# Paths (adjust if needed)
# =====================
BASE = "/work/data_assigments/DATA"
DATA_PATH = Path("/work/data_assigments/DATA/")

ALL_DATA_CSV        = os.path.join(BASE, "data_all.csv")
BID_META_CSV        = os.path.join(BASE, "bid_meta_df.csv")
EN_CLEAN_JSONL      = os.path.join(BASE, "all_english_entries_clean.jsonl")  # {bid, text}



CHATGPT_RELEASE = pd.Timestamp("2022-11-30")
POST_PERIOD_START = pd.Timestamp("2023-01-01")
TRANSITION_START = pd.Timestamp("2022-12-01")
TRANSITION_END = pd.Timestamp("2023-01-01")
ANALYSIS_START = pd.Timestamp("2018-01-01")

# Metadata fields required before constructing the downstream text-analysis sample.
# Grade is intentionally NOT required here; it is retained in the data and will
# be imposed only when the final common analytical sample is constructed.
REQUIRED_METADATA_COLUMNS = [
    "bid",
    "exam_code",
    "coursecode_clean",
    "AfleveringsTidspunkt",
    "take_home",
]

# Script 5 starts the cumulative sample-flow file. Downstream scripts can read
# this file, append their own restrictions, and overwrite it.
SAMPLE_FLOW_OUTPUT = DATA_PATH / "sample_flow.csv"
sample_flow_records = []


# In[3]:

# ================================================================
# HELPER FUNCTIONS
# ================================================================

def add_half_year_bins(data, date_col="AfleveringsTidspunkt"):
    """Add H1/H2 time bins to a copy of a dataframe."""
    result = data.copy()

    dates = pd.to_datetime(
        result[date_col],
        errors="coerce",
    )
    valid = dates.notna()

    period_start = pd.Series(
        pd.NaT,
        index=result.index,
        dtype="datetime64[ns]",
    )

    period_start.loc[valid] = pd.to_datetime(
        {
            "year": dates.loc[valid].dt.year,
            "month": np.where(
                dates.loc[valid].dt.month <= 6,
                1,
                7,
            ),
            "day": 1,
        }
    ).to_numpy()

    period = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )

    period.loc[valid] = (
        period_start.loc[valid].dt.year.astype(str)
        + " "
        + np.where(
            period_start.loc[valid].dt.month.eq(1),
            "H1",
            "H2",
        )
    )

    result["period_start"] = period_start
    result["period"] = period

    return result



def load_english_bids(jsonl_path: str) -> list:
    """Return unique BIDs from a JSONL file by reading it line by line."""
    bids = set()
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            bids.add(str(data["bid"]))
    return list(bids)


def print_sample_stats(label, data):
    """Print the main sample-size statistics for the current dataframe."""
    print(f"\n{label}")
    print(f"  Hand-ins: {len(data):,}")

    if "exam_code" in data.columns:
        print(f"  Exams: {data['exam_code'].nunique(dropna=True):,}")
    if "coursecode_clean" in data.columns:
        print(f"  Courses: {data['coursecode_clean'].nunique(dropna=True):,}")
    if "studerendeid" in data.columns:
        print(f"  Students: {data['studerendeid'].nunique(dropna=True):,}")


def append_sample_flow(stage, data, before_n=None):
    """Store one sample-construction checkpoint for the SI flow figure."""
    record = {
        "script": "5_clean_metadata",
        "stage": stage,
        "n_submissions": len(data),
        "n_removed": (before_n - len(data)) if before_n is not None else np.nan,
        "n_students": (
            data["studerendeid"].nunique(dropna=True)
            if "studerendeid" in data.columns
            else np.nan
        ),
        "n_courses": (
            data["coursecode_clean"].nunique(dropna=True)
            if "coursecode_clean" in data.columns
            else np.nan
        ),
        "n_exams": (
            data["exam_code"].nunique(dropna=True)
            if "exam_code" in data.columns
            else np.nan
        ),
        "n_pre": (
            int((data["after"] == 0).sum())
            if "after" in data.columns
            else np.nan
        ),
        "n_post": (
            int((data["after"] == 1).sum())
            if "after" in data.columns
            else np.nan
        ),
    }
    sample_flow_records.append(record)


def report_restriction(label, before_n, data, record_flow=False):
    """Print how many observations an existing cleaning step removed."""
    after_n = len(data)
    removed_n = before_n - after_n
    removed_pct = 100 * removed_n / before_n if before_n else 0

    print(f"\n🔹 {label}")
    print(f"  Before:  {before_n:,}")
    print(f"  After:   {after_n:,}")

    if removed_n >= 0:
        print(f"  Removed: {removed_n:,} ({removed_pct:.2f}%)")
    else:
        print(f"  Rows added by step: {-removed_n:,}")

    if "exam_code" in data.columns:
        print(f"  Exams remaining: {data['exam_code'].nunique(dropna=True):,}")
    if "coursecode_clean" in data.columns:
        print(f"  Courses remaining: {data['coursecode_clean'].nunique(dropna=True):,}")
    if "studerendeid" in data.columns:
        print(f"  Students remaining: {data['studerendeid'].nunique(dropna=True):,}")

    if record_flow:
        append_sample_flow(label, data, before_n=before_n)



# In[4]:


print("🔹 Loading metadata...")
all_data = pd.read_csv(ALL_DATA_CSV, dtype={"bid": str})
print_sample_stats("Loaded data_all.csv", all_data)

before_n = len(all_data)
all_data = all_data.loc[all_data["university"].isin(["KU", "RUC"])].copy()
report_restriction("Keep KU/RUC observations", before_n, all_data, record_flow=True)

# In[ ]:


bid_meta = pd.read_csv(BID_META_CSV, dtype={"bid": str})
print(f"\nLoaded bid_meta_df.csv: {len(bid_meta):,} rows")

before_n = len(bid_meta)
bid_meta = bid_meta.drop_duplicates("bid")
report_restriction("Drop duplicate BID rows in bid_meta", before_n, bid_meta)

before_n = len(bid_meta)
bid_meta = bid_meta.loc[bid_meta["len"] > 0].copy()
report_restriction("Keep bid_meta rows with len > 0", before_n, bid_meta)


# In[32]:


before_n = len(all_data)
final_meta = all_data.loc[all_data["bid"].isin(bid_meta["bid"])].copy()
report_restriction(
    "Keep hand-ins with a BID in bid_meta",
    before_n,
    final_meta,
    record_flow=True,
)


# In[33]:


print("🔹 Reading English BIDs from cleaned JSONL...")
eng_bids = load_english_bids(EN_CLEAN_JSONL)
final_meta["has_text"] = final_meta["bid"].astype(str).isin(eng_bids)
final_meta["has_text"] = final_meta["has_text"]*1

before_n = len(final_meta)
final_meta = final_meta.loc[final_meta["has_text"] == 1].copy()
report_restriction(
    "Keep hand-ins with cleaned English text",
    before_n,
    final_meta,
    record_flow=True,
)


# In[34]:


# Optional: keep some IDs as strings (mirrors your DK flow)
for col in ("Kursuskode"):
    if col in all_data.columns:
        final_meta[col] = all_data[col].astype(str)
# Rename columns for consistency
#final_meta = final_meta.rename({"Kursuskode": "examncode", "Kursusnavn": "examnname"}, axis=1)


# In[35]:


# Load mapping from BID to course ID
file_path = DATA_PATH / "besvarelsesid2courseid.js"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()
    bid_to_course = json.loads(content)
    bid2courseid = {str(k): v for k, v in bid_to_course.items() if v}  # Remove empty values


# In[36]:


# Map course code onto dataset from bid-to-course mapping
final_meta["coursecode"] = final_meta["bid"].map(bid2courseid)
final_meta["coursecode"] = final_meta["coursecode"].fillna(final_meta["Kursuskode"])
final_meta["coursecode_noE"] = final_meta["coursecode"].str.rstrip("E")
final_meta["coursecode_clean"] = final_meta["coursecode"].str.replace(r'(A|GB|\.0)$', '', regex=True)


# In[37]:


#clean up the meta datafile
before_n = len(final_meta)
reexam_counts = final_meta["is_reexam"].value_counts(dropna=False)
final_meta = final_meta.loc[final_meta["is_reexam"] == False].copy()
report_restriction("Remove re-exams", before_n, final_meta, record_flow=True)
print("  is_reexam values before restriction:")
print(reexam_counts.to_string())


# In[38]:


# Drop unnecessary or derived columns
cols = [
    'Unnamed: 0.1', 'Unnamed: 0', 'besvarelsesid', 'AnonymizeStudents',    'AnonymizeInvigilators',
    'cid','uid', 'nordic', 'dst', 'ar_AR', 'rarity_class', 'grade', 'Unnamed: 21',
    'muslim', 'north_eu', 'african', 'eu', 'socio_cluster_i',
    'nordic_lastname', 'muslim_lastname', 'north_eu_lastname', 'african_lastname',
    'eu_lastname', 'rarity_class_lastname', 'socio_cluster_i_lastname',
    'month', 'block', 'day', 't', 'days_since', 'days_bin',
    'studerende_firstseen', 'student_seen_before', 'student_seen_1year_before',
    'exam_size', 'exam_size_normed_max', 'ranked_size', 'ranked_size_normed_years',
    'is_first_try', 'first_try_percentage', 'anom_consistent', 'prev_ratio', 'prev_days',
    'next_ratio', 'next_days', 'is_reexam', 'is_bachelor', 'is_oral',
    'len', 'filesize', 'ending', 'c_d', 'month_int',
    '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021',
    '2022', '2023', '2024', '1', '2', '3', '4', '5', '6', '7', '8',
    '9', '10', '11', '12', 'prediction_linear', 'prediction',
    'Course_Dominant_Anonymized', 'same_as_dominant', 'has_censor','sex_ambigous',
    'Kursusnavn_indexed', 'Kursuskode_indexed', 'Sex_censor',
       'sex_ambigous_censor', 'eth_single_censor', 'nordic_censor',
       'dst_censor', 'ar_AR_censor', 'rarity_class_censor', 'Sex_eksaminator',
       'sex_ambigous_eksaminator', 'eth_single_eksaminator',
       'nordic_eksaminator', 'dst_eksaminator', 'ar_AR_eksaminator',
       'rarity_class_eksaminator', 'Same_Sex', 'Same_nordic',
       'Same_eth_single', 'Unnamed: 42',
       'muslim_censor', 'north_eu_censor', 'african_censor', 'eu_censor',
       'socio_cluster_i_censor', 'nordic_lastname_censor',
       'muslim_lastname_censor', 'north_eu_lastname_censor',
       'african_lastname_censor', 'eu_lastname_censor',
       'rarity_class_lastname_censor', 'eth_single_lastname_censor',
       'socio_cluster_i_lastname_censor', 'muslim_eksaminator',
       'north_eu_eksaminator', 'african_eksaminator', 'eu_eksaminator',
       'socio_cluster_i_eksaminator', 'nordic_lastname_eksaminator',
       'muslim_lastname_eksaminator', 'north_eu_lastname_eksaminator',
       'african_lastname_eksaminator', 'eu_lastname_eksaminator',
       'rarity_class_lastname_eksaminator', 'eth_single_lastname_eksaminator',
       'socio_cluster_i_lastname_eksaminator', 'Same_north_eu', 'Same_eu',
       'Same_muslim', "university"]
final_meta = final_meta.drop(columns=cols, errors='ignore')

print("🗑️  Dropped unused or irrelevant columns.")


# In[40]:


# In[2]:


# path to your saved file
inpath = Path(DATA_PATH) / "bid2examcode.json"

# load the dict back
with open(inpath, "r", encoding="utf-8") as f:
    bid2examcode_loaded = json.load(f)

print(f"📂 Loaded {len(bid2examcode_loaded)} mappings from {inpath}")


# In[3]:


final_meta["exam_code"] = final_meta["bid"].map(bid2examcode_loaded)


# In[4]:


# In[38]:
str_cols = [
    'studerendeid', 'karakter', 'bid',  'Sex',"censorid",
    'eth_single','Kursuskode', 'Kursusnavn', 'exam_code', 'coursecode','coursecode_noE'
]
final_meta[str_cols] = final_meta[str_cols].astype('string')


#datetime
final_meta["AfleveringsTidspunkt"] = pd.to_datetime(
    final_meta["AfleveringsTidspunkt"],
    errors="coerce",
)



# ================================================================
# 3. CONSTRUCT ANALYSIS-PERIOD VARIABLES
# ================================================================

final_meta["analysis_period"] = np.select(
    [
        final_meta["AfleveringsTidspunkt"] < CHATGPT_RELEASE,
        final_meta["AfleveringsTidspunkt"] >= POST_PERIOD_START,
    ],
    [
        "Before ChatGPT",
        "After ChatGPT",
    ],
    default="Transition month",
)

final_meta["after"] = final_meta["analysis_period"].map(
    {
        "Before ChatGPT": 0,
        "After ChatGPT": 1,
    }
)

final_meta["is_transition"] = final_meta[
    "AfleveringsTidspunkt"
].between(
    TRANSITION_START,
    TRANSITION_END,
    inclusive="left",
)

final_meta = add_half_year_bins(final_meta)


# In[7]:


course_dataset = pd.read_parquet(DATA_PATH / "english_coursedata.parquet")
cc = course_dataset["coursecode_clean"].unique()


# In[8]:


before_n = len(final_meta)
final_meta = final_meta.loc[final_meta["coursecode_clean"].isin(cc)].copy()
report_restriction(
    "Keep course codes represented in english_coursedata",
    before_n,
    final_meta,
    record_flow=True,
)

before_n = len(final_meta)
final_meta = final_meta.merge(course_dataset[["exam_code", "exam"]], on = "exam_code")
report_restriction(
    "Merge exam information from english_coursedata",
    before_n,
    final_meta,
    record_flow=True,
)


# In[9]:
GRADE_SOURCE = (
    "Karakter"
    if "Karakter" in final_meta.columns
    else "karakter"
)

final_meta["karakter_numeric"] = pd.to_numeric(
    final_meta[GRADE_SOURCE],
    errors="coerce",
)

GRADE_MAPPING = {
    -3: 0,
    0: 1,
    2: 2,
    4: 3,
    7: 4,
    10: 5,
    12: 6,
}


final_meta["karakter_0_6"] = (
    final_meta["karakter_numeric"]
    .map(GRADE_MAPPING)
)


# ## REad in exam information 
# 
# instead of using the classifyier below we should use the human labelled data 

# In[10]:
manual_exam_labels_path = "manual_labelling/examcode2manual_exam_label.json"

with open(manual_exam_labels_path, "r", encoding="utf-8") as f:
    examcode2label = json.load(f)


# ------------------------------------------------
# Assign exam information
# ------------------------------------------------

exam_type_labels = {
    1: "Written (take home)",
    2: "Written (invigilated)",
    3: "Pure Oral",
    4: "Other / Don’t Know",
}

# Human-labelled exam types
final_meta["exam_label"] = final_meta["exam_code"].map(examcode2label)


# ------------------------------------------------
# Add manually identified take-home exams
# ------------------------------------------------

keywords = [
    "hjemmeopgave",
    "hjemmeopg",
    "hj. opg.",
    "48 timers opgave",
    "online",
    "internship",
]

takehome_coursecodes = [
    "NFOB15003",
    "NFOK21004",
    "NDAB21002",
    "NIGK17001",
    "AØKK08334",
    "NNDB21000",
    "NDAB16009",
    "NMAK20003",
    "26546",
    "NFOK20002",
    "SMRM18005",
    "26536",
    "27215",
]

confirmed_takehome_names = [
    "Brewing Summer Course 7-5 ECTS - Rapport",
    "Videnskabsteori for matematiske fag - Skriftlig afl. 24 timer",
]

keyword_pattern = "|".join(
    re.escape(keyword)
    for keyword in keywords
)

recode_to_takehome = (
    final_meta["Kursusnavn"].str.contains(
        keyword_pattern,
        case=False,
        regex=True,
        na=False,
    )
    | final_meta["coursecode_clean"].isin(takehome_coursecodes)
    | final_meta["Kursusnavn"].isin(confirmed_takehome_names)
)

# Override these observations as written take-home exams
final_meta.loc[
    recode_to_takehome,
    "exam_label",
] = 1


# Add readable exam labels after all recoding is complete
final_meta["exam_label_verbose"] = (
    final_meta["exam_label"].map(exam_type_labels)
)




print(final_meta["exam_label_verbose"].value_counts())


final_meta["take_home"] = (final_meta["exam_label"] == 1).astype(int)
print(final_meta["take_home"].value_counts())


# In[18]:


before_n = len(final_meta)
missing_exam_label = final_meta["exam_label"].isna().sum()
final_meta = final_meta.dropna(subset=["exam_label"]).copy()
report_restriction("Drop hand-ins without an exam label", before_n, final_meta, record_flow=True)
print(f"  Missing exam_label before restriction: {missing_exam_label:,}")


# In[19]:

# ================================================================
# 7. APPLY COMMON ROW-LEVEL EXCLUSIONS
# ================================================================

missing_date = final_meta["AfleveringsTidspunkt"].isna().sum()
before_analysis_start = (
    final_meta["AfleveringsTidspunkt"].notna()
    & (final_meta["AfleveringsTidspunkt"] < ANALYSIS_START)
).sum()

before_n = len(final_meta)
final_meta = final_meta.loc[
    final_meta["AfleveringsTidspunkt"].notna()
    & (
        final_meta["AfleveringsTidspunkt"] >= ANALYSIS_START
    )
].copy()
report_restriction("Keep valid dates from 2018 onward", before_n, final_meta, record_flow=True)
print(f"  Missing date: {missing_date:,}")
print(f"  Dated before 2018-01-01: {before_analysis_start:,}")


transition_n = final_meta["is_transition"].sum()
before_n = len(final_meta)
final_meta = final_meta.loc[
    final_meta["after"].isin([0, 1])
    & ~final_meta["is_transition"]
].copy()
report_restriction(
    "Keep before/after periods and exclude transition month",
    before_n,
    final_meta,
    record_flow=True,
)
print(f"  Transition-month observations: {transition_n:,}")

# ================================================================
# 8. REQUIRE COMPLETE METADATA FOR SAMPLE CONSTRUCTION
# ================================================================

missing_required = (
    final_meta[REQUIRED_METADATA_COLUMNS]
    .isna()
    .sum()
)

before_n = len(final_meta)
final_meta = final_meta.dropna(
    subset=REQUIRED_METADATA_COLUMNS
).copy()
report_restriction(
    "Drop incomplete cases on required metadata",
    before_n,
    final_meta,
    record_flow=True,
)
print("  Missing values before restriction:")
if (missing_required > 0).any():
    print(missing_required[missing_required > 0].to_string())
else:
    print("  None")


# ================================================================
# 9. IDENTIFY COURSES THAT CHANGE WRITTEN EXAM TYPE
# ================================================================
# Identify these courses before removing invigilated examinations so that
# changes between take-home and invigilated formats remain observable.
# The flagged course IDs are removed after the take-home restriction below.

course_exam_types = (
    final_meta.loc[
        final_meta["after"].isin([0, 1])
        & final_meta["exam_label"].isin([1, 2])
    ]
    .groupby(["coursecode_clean", "after"])["exam_label"]
    .agg(lambda x: tuple(sorted(x.unique())))
    .unstack()
)

# Only compare courses with written examinations observed in both periods.
course_exam_types = course_exam_types.dropna(subset=[0, 1])

changing_exam_type_courses = course_exam_types.loc[
    course_exam_types[0] != course_exam_types[1]
].index

print(
    f"\nCourses changing written exam type before vs. after ChatGPT: "
    f"{len(changing_exam_type_courses):,}"
)


# ================================================================
# 10. RESTRICT TO TAKE-HOME EXAMINATIONS
# ================================================================

exam_type_counts_before = final_meta["exam_label_verbose"].value_counts(dropna=False)
before_n = len(final_meta)
final_meta = final_meta.loc[
    final_meta["exam_label"] == 1
].copy()
report_restriction(
    "Keep take-home examinations",
    before_n,
    final_meta,
    record_flow=True,
)
print("  Exam types before restriction:")
print(exam_type_counts_before.to_string())


# ================================================================
# 11. REMOVE COURSES THAT CHANGE WRITTEN EXAM TYPE
# ================================================================

before_n = len(final_meta)
final_meta = final_meta.loc[
    ~final_meta["coursecode_clean"].isin(changing_exam_type_courses)
].copy()
report_restriction(
    "Remove courses that change written exam type before/after ChatGPT",
    before_n,
    final_meta,
    record_flow=True,
)


# ================================================================
# 12. KEEP COURSES OBSERVED BOTH BEFORE AND AFTER CHATGPT
# ================================================================

course_period_coverage = (
    final_meta.groupby("coursecode_clean")["after"]
    .nunique()
)

matched_courses = (
    course_period_coverage.loc[
        course_period_coverage.eq(2)
    ]
    .index
)

single_period_courses = course_period_coverage.loc[
    course_period_coverage.eq(1)
].index

only_before_courses = [
    course
    for course in single_period_courses
    if (
        final_meta.loc[
            final_meta["coursecode_clean"] == course,
            "after",
        ] == 0
    ).all()
]

only_after_courses = [
    course
    for course in single_period_courses
    if (
        final_meta.loc[
            final_meta["coursecode_clean"] == course,
            "after",
        ] == 1
    ).all()
]

only_before_handins = final_meta["coursecode_clean"].isin(only_before_courses).sum()
only_after_handins = final_meta["coursecode_clean"].isin(only_after_courses).sum()

before_n = len(final_meta)
final_meta = final_meta.loc[
    final_meta["coursecode_clean"].isin(matched_courses)
].copy()
report_restriction(
    "Keep courses observed both before and after ChatGPT",
    before_n,
    final_meta,
    record_flow=True,
)
print(
    f"  Courses only before ChatGPT: {len(only_before_courses):,} "
    f"({only_before_handins:,} hand-ins)"
)
print(
    f"  Courses only after ChatGPT:  {len(only_after_courses):,} "
    f"({only_after_handins:,} hand-ins)"
)


# ================================================================
# 13. REMOVE COURSES WITH ONLY ONE TAKE-HOME HAND-IN
# ================================================================
# At this point the data contain take-home examinations only.

take_home_course_counts = (
    final_meta
    .groupby("coursecode_clean")
    .size()
)

singleton_take_home_courses = (
    take_home_course_counts.loc[
        take_home_course_counts == 1
    ]
    .index
)

print(
    f"\nCourses with only one take-home hand-in: "
    f"{len(singleton_take_home_courses):,}"
)

if len(singleton_take_home_courses) > 0:
    print(
        "  Courses: "
        + ", ".join(map(str, singleton_take_home_courses))
    )

before_n = len(final_meta)
final_meta = final_meta.loc[
    ~final_meta["coursecode_clean"].isin(singleton_take_home_courses)
].copy()
report_restriction(
    "Remove courses with only one take-home hand-in",
    before_n,
    final_meta,
    record_flow=True,
)


# ================================================================
# 14. FREEZE CLEAN TAKE-HOME METADATA SAMPLE
# ================================================================

final_meta = (
    final_meta
    .sort_values(
        [
            "AfleveringsTidspunkt",
            "exam_code",
            "bid",
        ]
    )
    .reset_index(drop=True)
)

if final_meta.empty:
    raise ValueError(
        "No observations remain in the clean take-home metadata sample."
    )

# Check that bid uniquely identifies each row
if final_meta["bid"].duplicated().any():
    duplicate_bids = final_meta.loc[
        final_meta["bid"].duplicated(keep=False),
        "bid",
    ].unique()

    print(f"\n⚠️ Found {len(duplicate_bids)} duplicated bid values.")
    print(duplicate_bids[:20])
else:
    print("\n✅ All bid values are unique.")


print_sample_stats("FINAL CLEAN METADATA SAMPLE", final_meta)

print("\nMissing values in final_meta")
print(final_meta.isna().sum().to_string())


# In[20]:


output_path = DATA_PATH / "english_metadata_clean.parquet"
final_meta.to_parquet(output_path)
print(f"\nSaved cleaned metadata to: {output_path}")

# Save sample-flow checkpoints for the SI sample-construction figure.
sample_flow_df = pd.DataFrame(sample_flow_records)
sample_flow_df.to_csv(SAMPLE_FLOW_OUTPUT, index=False)

print(f"Saved sample-flow checkpoints to: {SAMPLE_FLOW_OUTPUT}")
print("\nSample-flow checkpoints from script 5:")
print(sample_flow_df.to_string(index=False))
