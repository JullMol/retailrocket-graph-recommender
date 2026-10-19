# RetailGraph Recommender - Idea & Strategy Document

**Data Science Graphs 2026 · Midterm Exam (UTS) · Track: Graph Data Science & Recommender Systems**

This document captures the full reasoning behind **RetailGraph Recommender**, from problem framing and data wrangling through graph topology, Vector Space Modeling (VSM), bipartite projection decisions, and rigorous offline evaluation using Neo4j and Python.

---

## 1. Problem Statement

> In real-world e-commerce ecosystems characterized by massive catalogs and ultra-sparse user-item interactions (>99.99% sparsity), classical recommendation systems fail when evaluating future user intent due to temporal data leakage, cold-start fragmentation, and naive similarity projections. Retailers require an interpretable, graph-native recommendation and advertisement ranking engine that can mathematically model implicit behavioral signals, project bipartite interactions into actionable item-item similarity graphs, and provide verifiable ad recommendations without look-ahead bias.

Keywords deliberately embedded to connect with graph theory and recommender system evaluation criteria: *bipartite graph projection, Vector Space Model (VSM), implicit feedback hierarchy, temporal fold splitting, cosine neighborhood similarity, Hit@5, NDCG@5, Neo4j Cypher.*

**Empirical validation of this problem**: E-commerce interaction datasets (specifically the RetailRocket benchmark) exhibit extreme skewness where the vast majority of visitors interact with only 1–2 items before exiting, while items have a much broader distribution of interaction degrees. A naive random train-test split introduces severe *temporal look-ahead bias* by training on future interactions to predict past events. A strict, chronological temporal split is required to simulate genuine production deployment.

---

## 2. Target & Ecosystem

| Element | Detail |
|---|---|
| **Target Ecosystem** | RetailRocket e-commerce implicit feedback interaction platform (clicks, cart additions, purchases). |
| **Primary User (Target)** | Recurring visitors entering the testing window whose historical engagement traces must be translated into personalized recommendations. |
| **Secondary Stakeholder** | E-commerce advertising & merchandising desk requiring top-3 targeted item advertisements derived from a $k=5$ nearest-neighbor graph. |
| **Evaluation Audience** | Academic examiners & data science practitioners demanding fully reproducible graph modeling, verified Cypher queries, and statistical hypothesis testing. |

---

## 3. Issue Tree (MECE)

```
Root: Optimizing Graph-Based Top-K Recommendations under Extreme Interaction Sparsity
├── 1. Data Representation & Temporal Integrity
│   ├── 1.1 Temporal fold splitting (P1 Train vs P2 Test) to eliminate data leakage
│   ├── 1.2 Multi-modal implicit edge weighting (View=1, AddToCart=2, Bought=3)
│   └── 1.3 Preservation of multi-event parallel interactions in graph ingestion
├── 2. Topological Modeling & Bipartite Projection
│   ├── 2.1 Vector Space Modeling (VSM) duality (User-as-Document vs Item-as-Document)
│   ├── 2.2 Analytical candidate pair bounding vs computational complexity
│   └── 2.3 Neighborhood coverage under heavy-tailed degree distributions
└── 3. Inference, Materialization & Offline Evaluation
    ├── 3.1 Materializing :SIMILAR_TO weighted edges in Neo4j DBMS
    ├── 3.2 Top-K KNN retrieval with global popularity fallback for cold-starts
    └── 3.3 Paired statistical evaluation (Hit@5, NDCG@5 with 95% Bootstrap CIs)
```

**MECE Check**: The three branches represent three distinct, chronological, non-overlapping phases (data preparation, topological projection modeling, and evaluation/materialization) that together govern the lifecycle of a graph recommender system.

---

## 4. Three-Layer Analysis

### Macro (Domain & Temporal Context)
- **Temporal Non-Stationarity**: User preferences and product trends drift over time. Interaction logs cannot be randomly shuffled; training must precede testing chronologically ($P_1 < P_2$).
- **Implicit Feedback Hierarchy**: Users rarely leave explicit ratings. Instead, actions carry tiered intent: a `view` indicates mild interest (weight = 1), `addtocart` indicates strong intent (weight = 2), and `transaction` signifies conversion (weight = 3).
- **Recurrent Visitor Window**: Across a 9-day observation period, only $4.46\%$ of test visitors also appear in the training fold, setting the boundary for offline collaborative filtering evaluation.

### Micro (Topological & Graph Structure)
- **Bipartite Interaction Graph**: The natural data model is a directed bipartite multigraph $G = (U, I, E)$ with node partitions $U$ (`Visitor`) and $I$ (`Item`).
- **Degree Skewness & Asymmetry**: 
  - Mean user degree is **$1.46$** (median = $1.0$), reflecting extreme user fragmentation.
  - Mean item degree is **$2.73$** (median = $1.0$), with popular hubs capturing hundreds of co-occurrences.
- **Neighborhood Overlap**: Because user degrees are predominantly 1, user-user intersection $|N(u_1) \cap N(u_2)|$ is zero for the overwhelming majority of pairs, making user-user projection sparse and fragmented.

### Internal (Technical Stack & Constraints)
- **DBMS Architecture**: Neo4j 5.x Community / Desktop on Bolt port `7687` utilizing parameterized Cypher and batched `UNWIND` transactions.
- **Scientific Computing Environment**: Python 3.12, `neo4j` Python driver 5.28, `scipy.sparse` CSR matrices for memory-efficient cosine vector operations, and `numpy.random.default_rng` for deterministic reproducibility (Seed = `4084`).

---

## 5. Solution Design

**What**: An end-to-end graph-native recommender system that ingests raw e-commerce events into Neo4j, constructs Vector Space Models, compares bipartite projections, materializes high-confidence `:SIMILAR_TO` edges, and serves top-$N$ personalized item recommendations with top-3 advertisement placements.

**Objective**: Maximize recommendation ranking quality ($\text{NDCG@5}$) and recommendation coverage for returning visitors while maintaining sub-second inference latency.

**Why**: Item-Item projection provides **$3.67\times$ higher candidate neighborhood coverage** ($55\%$ vs $15\%$) and achieves a statistically significant improvement in ranking quality over User-User projection.

**How**:
1. Filter events into chronologically disjoint training ($P_1$) and testing ($P_2$) sets.
2. Ingest $P_1$ into Neo4j with unique node constraints and parallel edge types.
3. Construct dual VSM representations ($R$ and $R^T$) via SciPy CSR matrices.
4. Calculate sparse cosine similarity between item vectors, selecting the top-$k$ ($k=5$) nearest neighbors.
5. Materialize $75,220$ directed `:SIMILAR_TO` edges with score properties directly into Neo4j.
6. For a randomly selected target user (`visitorid = 1126007`), aggregate candidate similarity scores across historical profile items. If candidates are exhausted, apply a global popularity fallback.
7. Conduct point-in-time ranking evaluation against ground-truth testing interactions using Hit@5 and NDCG@5.

---

## 6. Methodology Summary (Execution Results)

### 6.1 Data Preparation & Temporal Splits ($P_1$ & $P_2$)

The interaction dataset is partitioned using strict chronological timestamps (UTC epoch milliseconds):

| Split / Fold | Start Date | End Date | Start Timestamp (ms) | End Timestamp (ms) | Total Events | Unique Visitors | Unique Items |
|---|---|---|---|---|---|---|---|
| **$P_1$ (Training)** | 2015-07-23 | 2015-07-31 | `1437609601209` | `1438387198354` | 217,823 | 119,895 | 64,086 |
| **$P_2$ (Testing)** | 2015-08-01 | 2015-08-09 | `1438387214822` | `1439164799534` | 159,238 | 93,338 | 55,272 |

- **Recurrent Visitor Overlap**: $4,161$ visitors ($4.46\%$ of test set) appeared in both folds.
- **Common Item Overlap**: $29,582$ items appeared in both folds.
- **Saved Files**: `data/processed/train_events.csv` and `data/processed/test_events.csv`.

---

### 6.2 Neo4j Subgraph Composition (Cypher Verification)

The training fold was imported into Neo4j with label uniqueness constraints (`visitorId`, `itemId`). The resulting composition was queried via the Python API:

| Category | Entity / Rel Type | Count | Cypher Query |
|---|---|---|---|
| **Nodes** | `:Visitor` | 119,895 | `MATCH (n) UNWIND labels(n) AS label RETURN label, count(*)` |
| **Nodes** | `:Item` | 64,086 | *(Same query as above)* |
| **Edges** | `:VIEWED` | 210,775 | `MATCH ()-[r]->() RETURN type(r) AS rel, count(*)` |
| **Edges** | `:ADDED_TO_CART` | 5,257 | *(Same query as above)* |
| **Edges** | `:BOUGHT` | 1,791 | *(Same query as above)* |
| **Total Interactions** | All Relationships | **217,823** | Sum of parallel events |
| **Unique Pairs** | `(Visitor, Item)` | **174,983** | Distinct bipartite pairs |

---

### 6.3 Vector Space Model (VSM) Duality

| Metric / Dimension | User VSM ($R$) | Item VSM ($R^T$) | Physical Interpretation |
|---|---|---|---|
| **Entity Representation** | Row = User, Col = Item | Row = Item, Col = User | User = document of items; Item = document of users |
| **Feature Space Dimension** | **$64,086$ dimensions** | **$119,895$ dimensions** | Item space is $1.87\times$ wider than user space |
| **Matrix Shape** | $(119,895 \times 64,086)$ | $(64,086 \times 119,895)$ | Exact transpose pair |
| **Non-Zero Entries (nnz)** | 174,983 | 174,983 | Identical non-zero interaction count |
| **Density** | $0.00002278$ | $0.00002278$ | Ultra-sparse interaction topology |
| **Sparsity (%)** | **$99.998\%$** | **$99.998\%$** | Exactly $99.9977\%$, correctly rounded to 3 decimals |
| **Mean Degree** | **$1.46$** | **$2.73$** | Items possess higher average co-occurrence connectivity |
| **Median Degree** | 1.0 | 1.0 | Extreme long-tail distribution |

---

### 6.4 Bipartite Projection Decision Analysis

The choice between User-User projection ($G_U$) and Item-Item projection ($G_I$) was evaluated on empirical efficiency, coverage, and ranking effectiveness:

| Evaluation Dimension | User-User Projection ($G_U$) | Item-Item Projection ($G_I$) | Advantage / Decision Basis |
|---|---|---|---|
| **Node Count ($N$)** | 119,895 | 64,086 | Item-item has $46.5\%$ fewer vertices |
| **Theoretical Dense RAM** | $115.0\text{ GB}$ | $32.9\text{ GB}$ | Theoretical dense bound ($71.4\%$ reduction) |
| **Candidate Pair Bound** | $1.10 \times 10^6$ | $0.86 \times 10^6$ | Ratio $1.27\times$ in favor of item-item |
| **Inference Time / Node** | $1.9 \times 10^{-5}\text{ s}$ | $5.5 \times 10^{-5}\text{ s}$ | Both complete in $< 1\text{ s}$ total runtime |
| **5-Candidate Coverage** | **$15.0\%$ of users** | **$55.0\%$ of users** | **Item-kNN provides $3.67\times$ superior coverage** |
| **Population Hit@5** | $0.0191$ $[0.0134, 0.0247]$ | $0.0299$ $[0.0230, 0.0368]$ | Overlapping marginal CIs |
| **Population NDCG@5** | $0.0106$ $[0.0071, 0.0141]$ | $0.0147$ $[0.0109, 0.0185]$ | Item-kNN superior |
| **Paired $\Delta\text{NDCG@5}$** | — | **$+0.0041$** | **95% Bootstrap CI: $[+0.00038, +0.00801]$ (Statistically Significant)** |

**Decision**: **Item-Item Projection** is selected. The decision is grounded in **paired statistical superiority ($\Delta\text{NDCG@5} > 0$)** and **actionable neighborhood coverage ($55\%$ vs $15\%$)**, not in execution time or RAM myths.

---

### 6.5 Materialization in Neo4j (State Before vs After)

- **Reference Item**: `itemId = 160053`
- **Cypher Top-5 Query**: Fully verified against Python sparse cosine dot products.
- **Top-5 Similar Items to 160053**:
  1. Item `258859` (Score: `0.217330`)
  2. Item `374981` (Score: `0.216422`)
  3. Item `250376` (Score: `0.170084`)
  4. Item `87892`  (Score: `0.146385`)
  5. Item `455772` (Score: `0.144266`)
- **Graph Transformation**:
  - `SIMILAR_TO` relationships before calculation: **0**
  - `SIMILAR_TO` relationships written after calculation: **75,220** (batched via `UNWIND` in 7.0 seconds).

---

### 6.6 Analytics: Random User Recommendation & Ad Selection

Using deterministic seed `4084`, the random selection process yielded:
- **Selected Target User**: `visitorid = 1126007` (Matrix row index = $95,994$).
- **Historical Training Profile**: 4 distinct items accessed (`3203`, `159285`, `284313`, `377804`), all with single views.
- **Top-3 Advertised Items ($k=5$ Item-kNN)**:
  1. **Item `18519`** (Score: **`0.447214`** · Source: **kNN** from item `159285`, 1 co-visitor, $\text{sim} = 1/\sqrt{5 \times 1}$)
  2. **Item `200909`** (Score: **`0.258199`** · Source: **kNN** from item `159285`, 1 co-visitor, $\text{sim} = 1/\sqrt{5 \times 3}$)
  3. **Item `149275`** (Score: **`0.000000`** · Source: **Popularity Fallback**, as other profile items lacked valid positive neighbors)

---

### 6.7 Evaluation: Hit@5 and NDCG@5

#### Single Random User Evaluation (`visitorid = 1126007`)
- **Testing Ground Truth**: Exactly 1 item (`itemid = 104801`).
- **Hit@5**: **0**
- **NDCG@5**: **0.0000**
- *Scientific Justification*: For a single user with 1 ground-truth target among $64,086$ items, Hit@5 is an all-or-nothing binary outcome ($0$ or $1$). Given an average population hit rate of $\approx 3\%$, the probability of a miss for any single randomly sampled user is $\approx 97\%$.

#### Population-Level Evaluation ($N = 2,309$ Eligible Recurrent Users)

To measure true system efficacy, the evaluation was conducted across the entire population of eligible recurrent visitors with 2,000 bootstrap iterations:

| Model Architecture | Population Hit@5 | 95% Bootstrap CI | Population NDCG@5 | 95% Bootstrap CI |
|---|---|---|---|---|
| **Popularity Baseline** | $0.0026$ ($0.26\%$) | $[0.0009, 0.0048]$ | $0.0005$ | $[0.0001, 0.0010]$ |
| **User-kNN ($k=5$)** | $0.0191$ ($1.91\%$) | $[0.0134, 0.0247]$ | $0.0106$ | $[0.0071, 0.0141]$ |
| **Item-kNN ($k=5$) [CHOSEN]** | **$0.0299$ ($2.99\%$)** | **$[0.0230, 0.0368]$** | **$0.0147$** | **$[0.0109, 0.0185]$** |

**Empirical Takeaway**: The chosen Item-kNN model outperforms the non-personalized popularity baseline by **$11.5\times$ in Hit@5** and **$29.4\times$ in NDCG@5**. Cell 71 in the notebook illustrates the exact mathematical mechanics on a hit user (`visitorid = 46121`), yielding $\text{Hit@5} = 1$ and $\text{NDCG@5} = 0.3066$.

---

## 7. Risk, Limitations & Mitigation

| Risk / Limitation | Reality & Nuance | Mitigation Strategy |
|---|---|---|
| **RAM Myth vs Sparse Reality** | The theoretical $115\text{ GB}$ vs $32.9\text{ GB}$ comparison applies only to dense matrices. | All production computations strictly utilize SciPy CSR sparse representations ($< 50\text{ MB}$ actual memory footprint). |
| **Marginal CI Overlap on Hit@5** | Marginal 95% CIs of Item-kNN ($[0.0230, 0.0368]$) and User-kNN ($[0.0134, 0.0247]$) overlap between $0.0230$ and $0.0247$. | Statistical hypothesis testing is conducted via **paired difference on NDCG@5** ($\Delta = +0.0041$, CI excludes zero), proving significance at $\alpha = 0.05$. |
| **Small Evaluation Sample Size** | The 9-day observation window produces an initial overlap of $4,161$ users ($4.46\%$), which shrinks to $2,309$ eligible users after deduplication. | Acknowledged transparently as an experimental boundary constraint; 2,000-resample bootstrap is employed to compute honest error bars. |
| **Single-User Zero Metric** | An uninitiated observer might interpret $\text{Hit@5} = 0$ on the random user as model failure. | Documented that single-user metrics on extreme catalogs are binary and noise-prone; population benchmarks ($N=2,309$) provide the true signal. |
| **Item Candidate Exhaustion** | Sparse users interacting with niche items frequently exhaust pure kNN neighbors. | Implemented a deterministic hybrid fallback to global item popularity (`fill_with_popularity`), ensuring all users receive exactly 5 recommendations. |
| **Domain Heuristics vs Computation** | The concept of "catalog stability" over time is a domain characteristic, not a computed variable. | Explicitly separated algorithmic findings (coverage, NDCG) from domain intuition (catalog vs visitor churn) in report narratives. |

---

## 8. Artifact & Deliverables Directory

All generated artifacts, exported tables, and visualization plots are persisted in the workspace:

- **Primary Notebook**: [UTS_Graf_Sains_Data.ipynb](file:///c:/Users/LENOVO/Documents/KULLYEAH/Semester%205/Graf%20Sains%20Data/UTS/UTS_Graf_Sains_Data.ipynb)
- **Processed Datasets**:
  - `data/processed/train_events.csv`
  - `data/processed/test_events.csv`
- **Output Tables**:
  - `outputs/tables/fold_summary.csv`
  - `outputs/tables/komposisi_simpul.csv`
  - `outputs/tables/komposisi_relasi.csv`
  - `outputs/tables/biaya_proyeksi.csv`
  - `outputs/tables/keputusan_proyeksi.csv`
  - `outputs/tables/evaluasi_model.csv`
  - `outputs/tables/before_after_relasi.csv`
- **Visualization Figures**:
  - `outputs/figures/vsm_user.png`
  - `outputs/figures/vsm_item.png`
  - `outputs/figures/projection_comparison.png`
  - `outputs/figures/projection_illustration.png`
  - `outputs/figures/before_after_graph.png`
- **Execution Run Metadata**:
  - `outputs/run_config.json`
  - `outputs/hasil_akhir.json`

---