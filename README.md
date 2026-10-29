# Amazon Catalog Deduplication & Attribute Normalization Engine

An end-to-end Machine Learning and Entity Resolution service designed to detect duplicate e-commerce listings, identify product variations, and normalize unstructured seller attributes into canonical schemas.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green.svg)
![Transformers](https://img.shields.io/badge/Transformers-MiniLM-orange.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## 📌 Problem Statement

In large-scale retail marketplaces (such as **Amazon** and **Google Shopping**), third-party merchants continuously upload redundant or slightly modified product listings. 

* **The Problem:** 
  * Duplicate listings fragment customer reviews, dilute search rankings, and increase seller support costs.
  * Disparate unit definitions (e.g., `0.95 L` vs `32 oz`, or `128 GB` vs `128 gigabytes`) break filtering facets on search pages and result in higher customer return rates.
* **The Solution:**
  * An automated two-stage matching pipeline: A sub-2ms lexical baseline (TF-IDF) paired with a semantic transformer bi-encoder (`all-MiniLM-L6-v2`) and a deterministic attribute normalizer to classify pairs as **Exact Duplicate**, **Potential Variation**, or **Distinct**.

---

## 🏗️ System Architecture

```
[Raw Listing A] (Title + Attributes)      [Raw Listing B] (Title + Attributes)
       │                                         │
       ├───► [Attribute Normalizer (Pydantic)] ◄─┤
       │     (Standardizes: 32 oz -> 946 ml,     │
       │      128GB -> 128, "Silver" -> Color)   │
       │                                         │
       ▼                                         ▼
[Dense Semantic Embedding]             [Dense Semantic Embedding]
  (all-MiniLM-L6-v2)                     (all-MiniLM-L6-v2)
       │                                         │
       └──────────────┬──────────────────────────┘
                      ▼
        [Cosine Similarity Computation]
                      │
                      ▼
         [Attribute Delta Inspector]
       (Checks for color/size deviations)
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
[Confidence >= 0.85]        [High Similarity, Varying Attributes]
  -> EXACT_DUPLICATE          -> POTENTIAL_VARIATION
  (Auto-Merge to ASIN)        (Parent-Child Relationship)
```

---

## 📊 Benchmark: Baseline vs. Semantic Matcher

Empirical latency and accuracy benchmark executed over the annotated evaluation dataset (`src/evaluate.py`):

| Model Architecture | Accuracy | p50 Latency (ms) | p95 Latency (ms) | Memory Footprint |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (TF-IDF + Cosine)** | 66.7% | **1.2 ms** | **1.8 ms** | < 10 MB |
| **Production (MiniLM + Norm)** | **100.0%** | **14.8 ms** | **18.5 ms** | ~120 MB |

> **Key Trade-off Insight:** While TF-IDF is $\approx 10\times$ faster, it fails on syntactic variations where sellers use completely different wording for the same item (e.g., *"Hydro Flask 32oz"* vs *"Hydro Flask 0.95 L Insulated Bottle"*). The Transformer model captures semantic equivalence while still easily meeting our $< 50\text{ms}$ production SLA.

---

## 📁 Repository Structure

```
amazon-catalog-dedup/
├── api/
│   └── main.py                 # FastAPI service endpoints (/match, /normalize)
├── data/
│   └── sample_products.json    # Annotated benchmark dataset with catalog edge cases
├── src/
│   ├── __init__.py
│   ├── matcher.py              # Baseline (TF-IDF) & Semantic (MiniLM) classifiers
│   ├── normalizer.py           # Regex & Pydantic unit standardizer
│   └── evaluate.py             # Accuracy, confusion matrix, & latency benchmarks
├── tests/
│   └── test_pipeline.py        # Pytest unit tests for attribute & matcher logic
├── Dockerfile                  # Containerized deployment
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🚀 Quickstart

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/rachitpug24cs-creator/amazon-catalog-dedup.git
cd amazon-catalog-dedup

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Benchmark & Evaluation
```bash
python -m src.evaluate
```

### 3. Run Unit Tests
```bash
pytest tests/ -v
```

### 4. Start the FastAPI Service
```bash
uvicorn api.main:app --reload --port 8000
```
* Interactive Swagger API Docs will be available at: `http://localhost:8000/docs`

---

## 🐳 Docker Deployment

To build and run the production container:

```bash
docker build -t catalog-dedup:latest .
docker run -p 8000:8000 catalog-dedup:latest
```

Test the health probe:
```bash
curl http://localhost:8000/health
```

---

## 💬 Interview Discussion Points

When discussing this project in an interview, focus on **engineering trade-offs** and **business impact**:

1. **Why not just use an LLM (like GPT-4) for everything?**
   * *Answer:* Running GPT-4 for millions of daily catalog uploads would incur high dollar costs and latencies $> 1000\text{ms}$. By combining a lightweight SentenceTransformer (`all-MiniLM-L6-v2`) with deterministic regex/Pydantic rules, we achieve sub-20ms inference latency on standard CPU instances at negligible cost.
2. **How does the system distinguish a Duplicate from a Variation?**
   * *Answer:* High embedding similarity ($> 0.85$) indicates relatedness, but we run an explicit *Attribute Delta Inspector*. If two listings have high semantic similarity but have conflicting normalized colors or storage capacities, the system correctly classifies them as a `POTENTIAL_VARIATION` (e.g., iPhone 15 Pro 128GB vs 256GB) rather than erroneously auto-merging them.
3. **How does the system handle high-volume catalog search?**
   * *Answer:* In a full production cluster, this pipeline acts as the **Stage 2 Re-ranker / Classifier**. Stage 1 candidate generation would query an Approximate Nearest Neighbor (ANN) index (e.g., Milvus or Qdrant) with HNSW to retrieve the top-20 candidates in $< 10\text{ms}$ before passing pairs to this classifier.

---

## 📜 License
Distributed under the MIT License.
