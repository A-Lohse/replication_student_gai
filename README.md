# The influence of generative artificial intelligence on student academic writing

Replication materials for:

**August Lohse and Felix Weiss, _The influence of generative artificial intelligence on student academic writing_.**

The study examines the early uptake of generative artificial intelligence in university examination submissions and associated changes in student writing and grades. The final analytical sample contains **44,374 English-language submissions from 25,364 students across 479 courses and 1,843 examinations** at the University of Copenhagen and Roskilde University, covering **3 January 2018 to 5 January 2024**.

## Repository structure

```text
replication/
├── code/
│   ├── 8_0_sample_restrictions.ipynb
│   ├── 8_1_gai_adoption_analysis.ipynb
│   ├── 8_2_linguistic_effects_of_gai.ipynb
│   └── 8_3_llm_decile_analysis.ipynb
│
├── preprocessing/
│   ├── 1_eng_lang_detect.py
│   ├── 2_clean_english_text.py
│   ├── 3_1_calc_complexity_measures.py
│   ├── 3_2_count_LLM_words.py
│   ├── 4_create_coursedata.py
│   ├── 5_clean_metadata.py
│   ├── 6_1_create_handin_data.py
│   ├── 7_1_embed_texts.py
│   ├── 7_2_calculate_embedding_distance.py
│   └── 7_3_finalize_analysis_sample.py
│
└── outputs/
    ├── figures/
    ├── tables/
    └── appendix/
```

## Reproducing the analyses

The four notebooks in `code/` reproduce the analyses, figures, and supplementary outputs from analysis data.

Run the notebooks **from the `replication/code/` directory** in the following order:

1. `8_0_sample_restrictions.ipynb`  
   Reproduces the analytical-sample flow diagram reported as Supplementary Fig. S1.

2. `8_1_gai_adoption_analysis.ipynb`  
   Reproduces the analyses of LLM-associated word use and the components of main-text Figure 1.

3. `8_2_linguistic_effects_of_gai.ipynb`  
   Reproduces the analyses of changes in writing characteristics, semantic similarity, and grades, including main-text Figure 2 and the corresponding supplementary analyses.

4. `8_3_llm_decile_analysis.ipynb`  
   Reproduces the heterogeneity analyses by LLM-associated word-rate decile, including main-text Figure 3 and the corresponding supplementary analyses.


and write generated files to:

```text
../outputs/figures/
../outputs/tables/
../outputs/appendix/
```

For example, after cloning the repository:

```bash
cd replication/code
jupyter lab
```

The notebooks can then be run sequentially from within Jupyter.


## Restricted source data

The data cannot be made publicly available because access is restricted under data-sharing agreements with the respective universities.

The scripts in `preprocessing/` document the complete processing pipeline used to construct the public analytical dataset, including:

- language detection;
- text cleaning;
- construction of linguistic-complexity measures;
- counting of LLM-associated words;
- course and examination metadata processing;
- sample construction;
- document embedding;
- semantic-similarity calculation; and
- construction and anonymization of the final analytical dataset.

All scripts are included for transparency but **cannot be run from the public repository alone**, because they require access to the restricted source data.

Examination format was manually classified during preprocessing. The underlying exam-level coding file is not included in the public repository because it is linked to restricted institutional metadata.

## Software

The analyses were conducted in Python. The principal software versions used for the final analyses were:

- Python 3.12.11
- pandas 2.3.2
- NumPy 2.3.3
- SciPy 1.18.1
- pyfixest 0.60.0
- Matplotlib 3.10.6

The notebooks also use Jupyter/IPython and `tqdm`.

## Reproducibility notes

All nonparametric bootstrap procedures use **1,000 replications**. A fixed random seed of **42** is used for sampling where applicable.

## Citation

Publication details and a persistent repository identifier will be added upon publication.
