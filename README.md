# 10-K Cybersecurity Disclosure Quality Scorer

This project measures the quality of cybersecurity and risk disclosures in SEC 10-K filings by analyzing **Item 1A (Risk Factors)** and **Item 1C (Cybersecurity, 2023+)** sections across a sample of **41 publicly listed companies** from **7 sectors** over fiscal years **2022–2025**.

---

## Folder Structure

```
sec-cyber-extraction/
│
├── extraction.py              # Step 1 — Pulls Item 1A & 1C from EDGAR → data/ + filings.parquet
├── filings.parquet            # Consolidated parquet: one row per company-year
├── requirements.txt           # Python dependencies
├── .env                       # API key (GEMINI_API_KEY) — not committed
│
├── data/                      # Extracted filing texts (one .txt per company-year)
│   ├── AAPL_2022.txt
│   ├── AAPL_2023.txt
│   ├── ...
│   └── WGO_2025.txt
│
├── code/                   # Step 2–7 — All scoring scripts
│   ├── length_analysis.py     #   Word-count analysis (Item 1A, 1C, combined)
│   ├── boilerplate_detector.py#   Boilerplate phrase ratio detection
│   ├── cosine_similarity.py   #   Year-over-year cosine similarity (TF-IDF)
│   ├── content_scoring.py     #   LLM-based content specificity scoring (Gemini)
│   ├── taxonomy_scoring.py    #   NIST CSF 2.0 keyword taxonomy + balance score
│   ├── specificity_scoring.py #   Composite specificity score (S)
│   ├── delta_scoring.py       #   Delta score (change effort indicator)
│   └── quality_scoring.py     #   Final quality score (Q = 0.8·S + 0.2·Δ)
│
├── visualizer/                # Visualization scripts (one per scoring module)
│   ├── length_analysis_visuals.py
│   ├── boilerplate_visuals.py
│   ├── similarity_visuals.py
│   ├── content_visuals.py
│   ├── balance_visuals.py
│   ├── function_weight_visuals.py
│   ├── quality_visuals.py
│   └── delta_visuals.py
│
├── results/                   # All scoring output files (CSV, Parquet, XLSX)
│   ├── length_results.csv / .parquet
│   ├── boilerplate_results.csv
│   ├── similarity_results.csv / .parquet
│   ├── content_scores.parquet / .xlsx
│   ├── nist_csf_scores.csv / .xlsx
│   ├── specificity_scores.csv
│   ├── delta_results.csv
│   ├── quality_results.csv
│   └── data_sample.xlsx       # Company metadata (sector, size, market cap)
│
└── visuals/                   # All generated figures (PNG)
    ├── length/
    ├── boilerplate/
    ├── similarity/
    ├── content/
    ├── balance/
    ├── function_weights/
    ├── quality/
    ├── specificity/
    └── delta/
```

---

## Pipeline Overview

The pipeline runs sequentially — each step reads the output of the previous one.

```
                        ┌─────────────────────────┐
                        │  1. extraction.py        │
                        │  Pull 10-K filings from  │
                        │  SEC EDGAR               │
                        └───────────┬──────────────┘
                                    │
                          data/*.txt + filings.parquet
                                    │
                        ┌───────────▼──────────────┐
                        │  2. length_analysis.py    │
                        │  Word counts per section  │
                        └───────────┬──────────────┘
                                    │
                         length_results.parquet
                                    │
                        ┌───────────▼──────────────┐
                        │  3. boilerplate_detector  │
                        │  Boilerplate phrase ratio  │
                        └───────────┬──────────────┘
                                    │
                       boilerplate_results.parquet
                                    │
                        ┌───────────▼──────────────┐
                        │  4. cosine_similarity.py  │
                        │  Year-over-year TF-IDF    │
                        │  similarity               │
                        └───────────┬──────────────┘
                                    │
                       similarity_results.parquet
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
  ┌───────────▼──────────┐  ┌──────▼───────────┐  ┌──────▼───────────┐
  │ 5. content_scoring   │  │ 6. taxonomy_     │  │  (parallel)      │
  │ LLM specificity      │  │    scoring.py    │  │                  │
  │ (Gemini API)         │  │ NIST CSF 2.0     │  │                  │
  └───────────┬──────────┘  │ keywords +       │  │                  │
              │             │ balance score    │  │                  │
              │             └──────┬───────────┘  │                  │
              │                    │              │                  │
      content_scores.parquet   nist_csf_scores    │                  │
              │                    │              │                  │
              └────────┬───────────┘              │                  │
                       │                          │                  │
           ┌───────────▼──────────────┐           │                  │
           │ 7. specificity_scoring   │           │                  │
           │ S = 0.6·Content + 0.4·  │           │                  │
           │     (1 − Boilerplate)    │           │                  │
           └───────────┬──────────────┘           │                  │
                       │                          │                  │
              specificity_scores.csv              │                  │
                       │                          │                  │
           ┌───────────▼──────────────┐           │                  │
           │  8. delta_scoring.py     │◄──────────┘                  │
           │  Δ = f(Specificity,      │                              │
           │       YoY Similarity)    │                              │
           └───────────┬──────────────┘                              │
                       │                                             │
                delta_results.csv                                    │
                       │                                             │
           ┌───────────▼──────────────┐                              │
           │  9. quality_scoring.py   │                              │
           │  Q = 0.8·S + 0.2·Δ      │                              │
           └───────────┬──────────────┘                              │
                       │                                             │
                quality_results.csv                                  │
                       │                                             │
           ┌───────────▼──────────────┐                              │
           │  10. Visualizers         │◄─────────────────────────────┘
           │  (visualizer/*.py)       │
           │  → visuals/*/*.png       │
           └──────────────────────────┘
```

---

## Step-by-Step Usage

### Prerequisites

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

For content scoring you also need a **Gemini API key** in `.env`:

```
GEMINI_API_KEY=your-key-here
```

### Step 1 — Extract filings from EDGAR

```bash
python extraction.py
```

- Pulls Item 1A & Item 1C for each company-year from SEC EDGAR
- Saves each filing as a `.txt` file in `data/` (e.g. `AAPL_2024.txt`)
- Writes a consolidated `filings.parquet` with all extractions

### Step 2 — Length analysis

```bash
python scoring/length_analysis.py
```

- Reads `filings.parquet` + `data_sample.xlsx` (company metadata)
- Computes word counts for Item 1A, Item 1C, and combined text
- **Output →** `results/length_results.parquet`, summary by size/sector

### Step 3 — Boilerplate detection

```bash
python scoring/boilerplate_detector.py
```

- Reads `length_results.parquet`
- Counts matches from a dictionary of ~50 boilerplate phrases
- **Output →** `results/boilerplate_results.parquet` (adds `boilerplate_count`, `boilerplate_ratio`)

### Step 4 — Cosine similarity

```bash
python scoring/cosine_similarity.py
```

- Reads `boilerplate_results.parquet`
- Computes year-over-year TF-IDF cosine similarity (4–6 n-grams)
- **Output →** `results/similarity_results.parquet` (adds `yoy_similarity`)

### Step 5 — Content scoring (LLM)

```bash
python code/content_scoring.py
```

- Reads `.txt` files from `data/`
- Sends each filing to **Gemini** for 6-category specificity scoring + boilerplate assessment
- **Output →** `results/content_scores.parquet` / `.xlsx` (binary scores + rationale per category)

### Step 6 — Taxonomy scoring (NIST CSF 2.0)

```bash
python code/taxonomy_scoring.py
```

- Reads `filings.parquet`
- Counts keyword hits for 6 NIST CSF functions (GV, ID, PR, DE, RS, RC)
- Computes function weights and a **balance score**
- **Output →** `results/nist_csf_scores.xlsx` / `.csv`

### Step 7 — Specificity scoring

```bash
python code/specificity_scoring.py
```

- Merges content scores + boilerplate ratios
- Computes **S = 0.6 × Content Score × 100 + 0.4 × (1 − Boilerplate Ratio) × 100**
- Classifies each filing as _High Specificity_ (≥ 60) or _Low Specificity_ (< 60)
- **Output →** `results/specificity_scores.csv` + visuals in `visuals/specificity/`

### Step 8 — Delta scoring

```bash
python code/delta_scoring.py
```

- Merges specificity scores with year-over-year similarity
- Computes a **delta multiplier** based on specificity and similarity categories
- **Output →** `results/delta_results.csv`

### Step 9 — Quality scoring

```bash
python code/quality_scoring.py
```

- Reads `delta_results.csv`
- Computes **Q = (0.8 × S + 0.2 × Δ) × 100**
- **Output →** `results/quality_results.csv`

### Step 10 — Generate visuals

```bash
python visualizer/length_analysis_visuals.py
python visualizer/boilerplate_visuals.py
python visualizer/similarity_visuals.py
python visualizer/content_visuals.py
python visualizer/balance_visuals.py
python visualizer/function_weight_visuals.py
python visualizer/quality_visuals.py
python visualizer/delta_visuals.py
```

Each visualizer reads its corresponding results from `results/` and saves figures as `.png` into the matching subfolder under `visuals/`.

---

## Output Summary

| Output location   | Contents                                                                             |
| ----------------- | ------------------------------------------------------------------------------------ |
| `data/`           | Raw extracted filing texts — one `.txt` per company-year (163 files)                 |
| `filings.parquet` | Consolidated extraction table (ticker, company, sector, year, has_1c, combined_text) |
| `results/`        | All intermediate and final scoring results (CSV, Parquet, XLSX)                      |
| `visuals/`        | All generated figures organized by analysis type (PNG)                               |

---

## Companies & Sectors

| Sector              | Tickers                                |
| ------------------- | -------------------------------------- |
| Consumer Goods      | CROX, ELF, MCFT, NKE, PEP, WGO         |
| Cybersecurity       | CRWD, PANW, PRGS, RPD, S, VRNS         |
| Finance             | HLI, LC, PSEC, UPST, V                 |
| Healthcare          | ELMD, JNJ, LLY, MODD, MOH, VKTX        |
| Retail & E-Commerce | AMZN, BOOT, ETSY, SFIX, UPWK           |
| Semiconductors      | AMD, CRUS, INTC, MXL, NVEC, POWI       |
| Technology          | AAPL, AMPL, GOOGL, GTLB, MSFT, SCSC, U |

---

## Notes

- **Item 1C** was introduced by the SEC in 2023, so 2022 filings contain Item 1A only. The `has_1c` column flags this.
- The `data/` folder serves as a raw backup. If the parquet ever needs to be rebuilt, it can be done from these files without re-hitting the EDGAR API.
- Content scoring requires a Gemini API key and is rate-limited to ~15 requests/minute (free tier).
- The boilerplate phrase dictionary can be extended in `code/boilerplate_detector.py` under `BOILERPLATE_PHRASES`.
- Some filings require manual URL overrides in `extraction.py` (see `MANUAL_OVERRIDES` dict).
