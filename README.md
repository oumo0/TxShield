<h1 align="center">🛡️ TxShield: Transaction-level RAG-enabled DeFi Attack Detection</h1>

<p align="center">
  <em>Reasoning over execution behavior with historical attack knowledge</em>
</p>
<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#results">Results</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#structure">Structure</a> ·
  <a href="#configuration">Configuration</a> ·
  <a href="#paper">Paper</a>
</p>

<p align="center">
  <a href="https://github.com/oumo0/TxShield/issues"><img src="https://img.shields.io/github/issues/oumo0/TxShield" alt="GitHub Issues"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/RAG-Historical_Attack_Knowledge-4F46E5" alt="RAG">
  <img src="https://img.shields.io/badge/Detection-LLM_%2B_XGBoost-7C3AED" alt="LLM and XGBoost">
  <img src="https://img.shields.io/badge/Paper-Under_Review_at_IEEE_TIFS-important" alt="Under review at IEEE TIFS">
</p>

<p align="center">
  <strong>92.93% F1 · 97.87% Precision · 88.46% Recall</strong>
  <br>
  <sub>Kimi-k3 backbone on DeFiAttackBench (728 transactions)</sub>
</p>

---

<a id="overview"></a>

## 🎯 Overview

DeFi attacks vary widely in implementation, yet many reuse a smaller set of behavioral patterns: manipulated prices, abnormal asset flows, reentrant call sequences, unauthorized state changes, and protocol-specific logic abuse. TxShield uses this observation to detect attacks at the **transaction execution trace** level.

Instead of relying only on fixed rules or an LLM's parametric memory, TxShield maintains a historical knowledge base of confirmed attacks. Each case combines an execution trace, an LLM-generated behavioral summary, and structured metadata. For a new transaction, the system retrieves semantically related cases and asks the LLM to reason over behavioral correspondence.

The final decision combines two complementary views:

- **Semantic reasoning:** the LLM interprets call sequences, state-changing behavior, fund movements, and historical analogies.
- **Risk-guided reasoning:** XGBoost estimates structured trace risk before LLM inference, controls whether retrieved cases are retained, and provides a non-binding probability hint.

This ordering is essential: **XGBoost may omit eligible retrieved cases and provide a risk hint, but it never directly assigns or overwrites the final label.**

### Why TxShield?

| Capability         | TxShield                                         | Conventional rule-based detection          |
| ------------------ | ------------------------------------------------ | ------------------------------------------ |
| Analysis target    | Runtime transaction behavior                     | Predefined vulnerability signatures        |
| Knowledge          | 209 historical entries in the paper's experimental knowledge base | Static rules and manually encoded patterns |
| Generalization     | Behavioral analogy across protocols              | Strongest on already modeled mechanisms    |
| Reasoning          | RAG or zero-shot, selected dynamically           | Fixed matching or symbolic constraints     |
| Numerical guidance | XGBoost risk-guided context and prompting        | Tool-specific alert logic                  |
| End-to-end latency | 11.82–37.54 s across evaluated backbones         | Varies by tool and analysis paradigm       |

**Core idea:** TxShield reasons about *attack behavior*, not only vulnerability signatures. It asks: **"Have we seen this behavior before?"**

<a id="architecture"></a>

## 🏗️ Architecture

<p align="center">
  <img src="assets/flowchart.png" alt="TxShield risk-guided architecture" width="900">
</p>


TxShield contains two cooperating components.

### 📚 Historical Knowledge Construction

Historical exploit transactions are collected and converted into execution traces. The traces are minimized to preserve security-relevant control flow, asset flow, and state-dependent operations while reducing redundant context. An LLM summarizes each attack's behavior, and the summaries are embedded with `jina-embeddings-v3` and stored in Chroma with trace evidence and metadata.

The knowledge base described in the paper contains **209 historical entries** and can be expanded as newly confirmed incidents become available.

### 🔍 RAG-based Decision Engine

1. **Preprocess the transaction.** Normalize and prune the execution trace before it is submitted to the analysis entry point.
2. **Retrieve Top-5 cases.** Search the historical Chroma collection using the same Jina embedding model used during indexing.
3. **Apply the relevance threshold.** Convert Chroma's squared L2 scores to L2 distances, then retain cases at distance ≤ 0.15 (the paper's similarity threshold `τ = 0.85`). If no cases remain, use zero-shot LLM analysis; absence of similar attacks is not a benign verdict.
4. **Estimate transaction risk.** When eligible cases remain, XGBoost derives a structured-feature attack probability for the transaction.
5. **Select the reasoning strategy.** Retain relevant cases only when the risk probability meets the context threshold (0.5 by default); otherwise use conservative zero-shot reasoning. The score is a non-binding LLM hint, not a verdict.
6. **Obtain the LLM decision.** The LLM produces the final `Attack` or `Benign` label together with its behavioral explanation.

All supported backends receive the same paper-aligned RAG or zero-shot prompt. When eligible cases exist, XGBoost is evaluated before LLM inference: its risk estimate controls the use of those cases and is included as auxiliary evidence. It never overrides the LLM's final label.

<a id="results"></a>

## 📊 Results

### DeFiAttackBench

DeFiAttackBench contains **364 attack transactions** from DeFiHackLabs and **364 benign transactions** sampled from normal activity in affected protocols. In the paper's experimental split, the attacks span January 2021 to January 2025, cover more than 110 labeled exploit types across chains including Ethereum, BSC, and Arbitrum, and the 728 evaluation transactions are disjoint from both XGBoost training and knowledge-base construction. The main comparison reported by the paper is:

| Method       | Evaluated |   Accuracy |  Precision |     Recall |         F1 |
| ------------ | --------: | ---------: | ---------: | ---------: | ---------: |
| TxSpector    |       450 |      54.4% |      50.0% |      57.1% |      53.3% |
| TrxGNNBERT   |       728 |      75.9% |      76.0% |      77.3% |      76.6% |
| Blockscan    |       728 |      83.4% |      95.5% |      70.1% |      81.0% |
| XGBoost only |       728 |     86.13% |     84.33% |     88.74% |     86.48% |
| **TxShield** |   **728** | **93.27%** | **97.87%** | **88.46%** | **92.93%** |

TxShield reduces false positives from **60 to 7** relative to direct XGBoost classification, while adding one false negative. The combined system uses historical behavioral evidence, XGBoost-guided context selection, and LLM reasoning rather than treating either component as sufficient on its own.

<sub>TxSpector processed 450 of 728 transactions.</sub>

### PriceAttackBench

On the 91 evaluated price-manipulation attacks from DeFiScope's D1 dataset:

| Method       | Detected attacks | Detection rate |
| ------------ | ---------------: | -------------: |
| DeFiTainter  |          32 / 91 |         35.16% |
| DeFiRanger   |          48 / 91 |         52.75% |
| DeFort       |          50 / 91 |         54.95% |
| DeFiScope    |          73 / 91 |         80.22% |
| **TxShield** |      **86 / 91** |     **94.51%** |

### 🎯 Attack Type Performance

Table VIII of the paper reports category-level detection rates and relates them to the behavioral evidence available in execution traces:

| Attack type                          | Detection rate | Difficulty    |
| ------------------------------------ | -------------: | ------------- |
| Reentrancy                           |          96.2% | 🟢 Low         |
| Price & Oracle Manipulation          |          93.2% | 🟢 Low         |
| Flash-loan Price Manipulation        |          91.7% | 🟢 Low         |
| Business Logic Flaws                 |          89.6% | 🟡 Medium      |
| Liquidity & Migration Exploits       |          87.5% | 🟡 Medium      |
| Arbitrary Call / External Call       |          86.7% | 🟡 Medium      |
| Access Control                       |          79.3% | 🟡 Medium      |
| Fake Market / MEV / Sandwich Attacks |          75.0% | 🔴 High        |
| Signature & Validation Issues        |          57.1% | 🔴 High        |
| Cryptographic Exploits               |           <50% | 🔴 Fundamental |

<sub>Reentrancy includes two cross-contract cases. Category-level rates are computed independently of the aggregate default-backbone metrics.</sub>

### Efficiency

The paper reports that retrieval itself completes in under 100 ms; total latency is dominated by LLM inference and network communication. Across the evaluated backbones, average end-to-end latency ranges from **11.82 s** with Llama-3.1-8B-Instruct to **37.54 s** with Qwen3-8B; GPT-4.1-mini averages **15.02 s**. Reported API cost is total evaluation cost, not cost per transaction: approximately **$0.26 for 728 transactions** with the evaluated Llama configuration and **$3.86** with GPT-4.1-mini.

> Results above are taken from the paper's experiments and are not recomputed automatically during installation. Costs vary with provider pricing and model availability.

## ✨ Key Features

- **Transaction-level analysis** of nested EVM call traces and value movement.
- **Two-stage retrieval** with Top-5 search followed by relevance filtering.
- **Adaptive prompting** that switches between RAG and zero-shot reasoning.
- **Unified paper-aligned prompting** across official APIs, Ollama, and local Transformers, using the manuscript's Figure 2 and Figure 3 templates.
- **Evidence-bound RAG reasoning:** retrieved cases guide behavioral comparison but do not by themselves establish maliciousness.
- **Expandable historical memory** built from confirmed attack behavior.
- **Risk-guided retrieval and prompting:** XGBoost risk estimates select retrieved context and provide auxiliary LLM evidence; the LLM produces the final label.
- **Compatible model backends:** official APIs, Ollama, and local Transformers.
- **Stable output contract** retaining both `llm_prediction` and `final_prediction` with XGBoost metadata.
- **Deep-trace feature extraction** using iterative traversal, preserving the 14-item candidate order and deriving the paper's 10 model inputs.

<a id="quick-start"></a>

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/oumo0/TxShield.git
cd TxShield

python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[rag]"
```

Add the backend extra only when it is needed:

```bash
pip install -e ".[rag,ollama]"     # Ollama server
pip install -e ".[rag,local-llm]"  # Hugging Face Transformers
```

Copy the environment template:

```bash
# Linux/macOS
cp .env.example .env

# Windows PowerShell
Copy-Item .env.example .env
```

Fill only the paths and credentials required by the backend you intend to use.
The default API provider is Kimi, matching the manuscript's default backbone.
Provider model availability and account access must be checked in your environment.

Run commands from the project root. Relative paths are resolved against the
current working directory. Input traces must contain one nonempty root list:
`function_call`, `calls`, or `logs.calls`. Child calls use `internal_calls` or
`calls`; a node must not supply both with nonempty children. For example:

```json
{"calls": [{"type": "CALL", "function": "transfer", "contract": "0xabc", "value": "0", "internal_calls": []}]}
```

Prepare separate raw training and test directories. Label each JSON filename
with `_attack.json` or `_benign.json`, and keep the transaction sets disjoint:

```bash
txshield preprocess --config configs/preprocessing.yaml --input data/raw/train --output data/processed/train
txshield preprocess --config configs/preprocessing.yaml --input data/raw/test --output data/processed/test
txshield train --config configs/training.yaml
```

These commands require external trace data; the benchmark CSVs are transaction
lists and are not training-ready execution traces. Knowledge-base construction
and the metadata/vector artifacts must also be prepared separately.

### Analyze a transaction

`analyze_contract()` accepts a normalized transaction execution trace. For raw
call logs, run the repository's preprocessing workflow first with
`configs/preprocessing.yaml`; the public analysis entry does not invoke
directory-level preprocessing automatically.

```python
from txshield.contract_analyzer.main import analyze_contract

result = analyze_contract(
    query="Analyze this DeFi transaction for attack behavior.",
    system_input_path="path/to/transaction.json",
)

if result.get("error") or result.get("final_prediction") == "Unknown":
    raise RuntimeError(result.get("error") or "No valid Attack/Benign label was returned.")

print("LLM prediction:", result["llm_prediction"])
print("Final prediction:", result["final_prediction"])
print("Used RAG:", result["used_rag"])
print("Explanation:", result["answer"])
```

Check `error`, `retrieval_error`, `xgb_error`, and `final_prediction == "Unknown"`
before treating a result as a successful complete-pipeline prediction. Retrieval
or scoring failures may produce a degraded result rather than a full-system run.

A transaction JSON string can be supplied without creating a file:

```python
from txshield.contract_analyzer.main import analyze_contract

transaction_json = '{"calls": [{"function": "transfer", "contract": "0xabc", "value": "0"}]}'

result = analyze_contract(
    query="Analyze this DeFi transaction for attack behavior.",
    system_input_text=transaction_json,
)
```

Successful analysis responses include the following fields. Failed analyses
return `final_prediction: "Unknown"` and an `error`; optional metadata may be absent.

| Field                    | Description                                      |
| ------------------------ | ------------------------------------------------ |
| `answer`                 | Original LLM decision and behavioral explanation |
| `llm_prediction`         | `Attack` or `Benign` parsed from the LLM response; `Unknown` if no valid label is available |
| `final_prediction`       | Final LLM label, or `Unknown` for an invalid or failed analysis |
| `used_rag`               | Whether retrieved historical context was used    |
| `similarity_info`        | L2 distances (smaller means closer) for eligible candidates, including those later omitted by risk guidance; `all_similarities` contains `1 - distance` |
| `xgb_applied`            | Whether an XGBoost risk score was available      |
| `xgb_attack_probability` | XGBoost score when available                     |
| `xgb_threshold`          | Context-selection threshold used for risk guidance |
| `xgb_context_retained`   | Whether eligible retrieved cases were retained by risk guidance |
| `xgb_decision`           | `retain_context`, `omit_context`, or score availability status |
| `error`                  | Analysis failure, when present |
| `retrieval_error`        | Retrieval or knowledge-metadata failure, when present |
| `xgb_error`              | XGBoost scoring failure, when present |

<a id="structure"></a>

## 📁 Project Structure

```text
TxShield/
├── assets/
│   └── flowchart.png               # architecture figure from the manuscript
├── configs/
│   ├── preprocessing.yaml          # trace preprocessing configuration
│   └── training.yaml               # paper/grid-search XGBoost settings
├── Data/
│   ├── DeFiAttackBench.csv         # benchmark transaction list and labels
│   └── PriceAttackBench.csv        # price-attack benchmark metadata
├── scripts/
│   ├── preprocess_dataset.py        # YAML-driven trace preprocessing
│   ├── train_model.py               # paper-mode or grid-search XGBoost training
│   └── evaluate_model.py            # standalone XGBoost evaluation utility
├── src/txshield/
│   ├── contract_analyzer/
│   │   ├── backends/               # official API, Ollama, local model adapters
│   │   ├── core/                   # embeddings, retrieval, paper-aligned prompts, RAG chain
│   │   ├── models/                 # XGBoost risk-scoring integration
│   │   ├── pipeline/               # label parsing and final LLM decision metadata
│   │   ├── utils/                  # legacy feature-extractor compatibility
│   │   ├── config.py               # runtime environment settings
│   │   └── main.py                 # analyze_contract public entry point
│   ├── detection/                  # XGBoost features, training, inference, and evaluation
│   ├── preprocessing/              # trace cleaning and pruning
│   ├── config/                     # YAML configuration loading
│   ├── common/                     # trace validation, file and logging utilities
│   ├── api/                        # optional standalone XGBoost API adapter
│   └── cli.py                      # preprocessing and training CLI
├── .env.example                    # runtime configuration template
├── .gitignore                      # local data, models and generated-file exclusions
├── pyproject.toml                  # package and optional dependencies
├── requirements.txt                # core numerical/preprocessing dependencies
└── README.md
```

The `data/raw/train`, `data/raw/test`, `data/processed/train`, and
`data/processed/test` paths in the commands are local dataset directories.
The `artifacts/` paths hold locally prepared or generated research artifacts.
These directories are not included in the source tree above. The example
train/test split refers to transaction data used for model training and
evaluation.

The package keeps the paper-facing detection workflow in `contract_analyzer/`, while `detection/` and `preprocessing/` provide reusable training and trace-preparation components. `core/message_manager.py` is the single prompt source for all supported LLM backends. The documented `analyze_contract` interface is intentionally stable; its `query` argument remains accepted for compatibility, while the manuscript's fixed detection task defines the actual Figure 2/3 prompt.

<a id="configuration"></a>

## ⚙️ Configuration

TxShield separates runtime configuration from reproducible preprocessing and training settings:

- `.env` contains local paths, backend selection, and credentials.
- `configs/preprocessing.yaml` contains trace-minimization settings.
- `configs/training.yaml` contains XGBoost training settings.
- Explicit command-line arguments override YAML values.

The preprocessing YAML is used by the preprocessing workflow. The runtime
`analyze_contract()` interface expects the resulting normalized trace rather
than running that directory-level workflow internally.

### Environment variables

Copy `.env.example` to `.env`. The template is intentionally compact: configure only the selected backend and the local artifacts used by your run.

The online paper pipeline requires an existing Jina-compatible Chroma database,
its knowledge metadata CSV, and a trained XGBoost model.

The default Chroma collection is `langchain`; its directory and nonempty L2
collection must already exist. The metadata CSV requires `id` and
`description_Security` columns. IDs are kept as strings; UTF-8 is preferred,
with MacRoman retained as a fallback for older artifacts.

The main runtime settings are:

```dotenv
TXSHIELD_VECTOR_DIR=artifacts/vector_store
TXSHIELD_CSV_PATH=artifacts/knowledge/knowledge.csv
TXSHIELD_XGB_MODEL=artifacts/models/xgb_final_model.pkl
TXSHIELD_XGB_CONTEXT_THRESHOLD=0.5

JINA_API_KEY=
TXSHIELD_EMBEDDING_MODEL=jina-embeddings-v3
TXSHIELD_SIMILARITY_THRESHOLD=0.85
TXSHIELD_RETRIEVAL_K=5
```

The retrieval quality gate follows the paper's formula `similarity = 1 - L2_distance`.
Chroma's `l2` scores are squared distances: a score of `0.0225` corresponds to
L2 distance `0.15` and similarity `0.85`. The runtime takes the square root before
filtering, with a small float32 rounding tolerance at the inclusive boundary.
Collections configured for cosine or inner-product distance are rejected rather
than interpreted as L2. The legacy `TXSHIELD_DISTANCE_THRESHOLD`, if set, takes
precedence and is interpreted as an L2 distance, not a squared distance.
Changing the retrieval threshold or `TXSHIELD_XGB_CONTEXT_THRESHOLD` creates an
experimental configuration rather than reproducing the reported paper pipeline.

When no cases pass retrieval filtering, XGBoost scoring is skipped and the LLM
uses the conservative zero-shot prompt. When XGBoost discards eligible context,
its already-computed probability remains as an auxiliary hint in that prompt.
Every successful classification is still made by the LLM.

Training and online analysis use `artifacts/models/xgb_final_model.pkl` by default.
Files ending in `.pkl` or `.pickle` are saved and loaded with Python pickle;
JSON/UBJ models remain supported when explicitly selected. Changing a filename
extension does not convert an existing model. Only load pickle models from trusted
sources, using Python, XGBoost, and scikit-learn versions compatible with the
training environment.

The standalone XGBoost evaluation command accepts the same pickle artifact:

```bash
python scripts/evaluate_model.py --model artifacts/models/xgb_final_model.pkl --threshold artifacts/thresholds/best_threshold.txt --test data/processed/test
```

This script evaluates standalone XGBoost predictions. It does not execute the
full RAG/LLM benchmark or regenerate the manuscript's TxShield result tables.

The saved classification threshold is used for standalone evaluation; online
context selection separately uses `TXSHIELD_XGB_CONTEXT_THRESHOLD=0.5`.

### LLM backends

Select exactly one runtime mode:

| Mode         | Configuration              | Use case                            |
| ------------ | -------------------------- | ----------------------------------- |
| Official API | `TXSHIELD_LLM_MODE=api`    | OpenAI, DeepSeek, or Moonshot/Kimi  |
| Ollama       | `TXSHIELD_LLM_MODE=ollama` | Local model served by Ollama        |
| Transformers | `TXSHIELD_LLM_MODE=local`  | In-process local Hugging Face model |

Official API example:

```dotenv
TXSHIELD_LLM_MODE=api
TXSHIELD_API_PROVIDER=kimi
TXSHIELD_LLM_MODEL=
TXSHIELD_API_BASE_URL=

OPENAI_API_KEY=
DEEPSEEK_API_KEY=
MOONSHOT_API_KEY=
```

Supported official providers are `openai`, `deepseek`, and `kimi`. Kimi uses Moonshot's official China endpoint, `https://api.moonshot.cn/v1`, by default.

The paper uses `Kimi-k3` as its default backbone for DeFiAttackBench and the component ablations. Set `TXSHIELD_API_PROVIDER=kimi` and leave `TXSHIELD_LLM_MODEL` empty (or set it explicitly to `kimi-k3`) to select the corresponding repository backend.

Local backend examples:

```dotenv
# Ollama
TXSHIELD_LLM_MODE=ollama
TXSHIELD_OLLAMA_URL=http://localhost:11434
TXSHIELD_OLLAMA_MODEL=qwen2.5:3b

# Hugging Face Transformers
TXSHIELD_LLM_MODE=local
TXSHIELD_LOCAL_MODEL=path-or-model-id
TXSHIELD_MAX_NEW_TOKENS=512
```

### Paper-aligned XGBoost configuration

`training_mode: paper` locks the seven parameters reported by the paper and rejects conflicting values:

```yaml
max_depth: 3
learning_rate: 0.02
min_child_weight: 1
subsample: 0.9
colsample_bytree: 0.7
reg_alpha: 0
reg_lambda: 0.5
```

Only these seven model parameters are specified in the manuscript. Defaults
such as 500 estimators, seed 42, validation size 0.2, pruning depth 12, and
argument length 200 are repository implementation settings. Matching the seven
parameters alone does not establish reproduction of the reported experiments.
Embedding input is currently limited to the first 8192 characters; the local
Transformers backend defaults to 512 output tokens. These limits are also
implementation settings rather than manuscript-specified experiment values.

The runtime derives the paper's ten model inputs from the established 14-item
candidate vector in a stable order. It retains call count, event count, maximum
call depth, unique contract count, transfer/approve/flashloan indicators, total
call-node value, unique signatures, and unknown signatures. Total call-node
value is transformed with `log1p`; the context-selection threshold is `0.5` by
default and may be set with `TXSHIELD_XGB_CONTEXT_THRESHOLD`.



## 📦 Datasets and artifacts

This repository includes the benchmark transaction lists:

- `Data/DeFiAttackBench.csv` (728 evaluation transactions: 364 attack and 364 benign)
- `Data/PriceAttackBench.csv` (95 collected candidates; the paper evaluates the 91 transactions that could be parsed)

The paper's separate balanced XGBoost training set contains 420 transactions. That training set, full execution traces, the 209-entry knowledge base, local Chroma data, API credentials, and trained model files are external research artifacts and are intentionally not committed by default. Their provenance and redistribution rights should be verified before a public artifact release.

## ⚠️ Limitations and future work

TxShield is strongest when malicious intent is observable in execution traces and a new incident shares behavioral structure with known attacks. Detection is more difficult when intent depends on cryptographic validity, transaction ordering, long-range state, or dispersed multi-contract evidence.

Current engineering limitations include the absence of a complete public command for rebuilding the 209-entry knowledge base from raw exploit traces, the requirement to preprocess raw traces before calling `analyze_contract()`, and the need to verify retrieval-score consistency when migrating the existing Chroma collection.

Algorithm 1 selects the deepest target occurrence, while Section IV-A mentions
the first occurrence. This implementation follows Algorithm 1 and applies it
independently to each supplied root tree. Figure 3's prompt is retained as written
in the manuscript; its example concerning state changes before external calls
needs clarification against the intended security semantics. The current source
does not include original model weights or evidence that its outputs reproduce
the paper's tables; reported metrics above remain manuscript results.

Future directions described by the paper include:

1. formal verification and symbolic-analysis integration;
2. richer cross-contract call-graph and state-flow reasoning;
3. automated, continuously validated knowledge-base expansion.

<details>
<summary><strong>📈 Additional paper results</strong></summary>

### Complete risk-guided backbone results

Table V reports three representative backbones using the complete TxShield pipeline, with 728 evaluated transactions for each backbone:

| Backbone        |      TP |    FP |      TN |     FN |   Accuracy |  Precision |     Recall |         F1 |
| --------------- | ------: | ----: | ------: | -----: | ---------: | ---------: | ---------: | ---------: |
| **Kimi-k3**     | **322** | **7** | **357** | **42** | **93.27%** | **97.87%** | **88.46%** | **92.93%** |
| DeepSeek-v4-pro |     325 |    24 |     340 |     39 |     91.35% |     93.12% |     89.29% |     91.16% |
| GPT-5.6-luna    |     318 |    29 |     335 |     46 |     89.70% |     91.64% |     87.36% |     89.45% |

Kimi-k3 provides the highest precision and F1 of these backbones, while DeepSeek-v4-pro provides the highest recall.

### Component ablation

Table IV isolates the contribution of retrieval and XGBoost. It is included here as supporting analysis rather than as the primary benchmark table.

| Configuration                           |    N | TP/FP/TN/FN    | Accuracy | Precision | Recall |     F1 |
| --------------------------------------- | ---: | -------------- | -------: | --------: | -----: | -----: |
| Full system                             |  728 | 322/7/357/42   |   93.27% |    97.87% | 88.46% | 92.93% |
| Without XGBoost guidance                |  728 | 295/98/266/69  |   77.06% |    75.06% | 81.04% | 77.94% |
| Without XGBoost guidance and two-stage retrieval | 728 | 315/110/254/49 | 78.16% | 74.12% | 86.54% | 79.85% |
| Without RAG                             |  726 | 278/98/266/84  |   74.93% |    73.94% | 76.80% | 75.34% |

</details>

## 🤝 Contributing

Contributions are welcome, especially for new historical attack patterns, documentation improvements, backend compatibility, and reproducible artifact packaging. Please open an issue before proposing a change that alters the paper pipeline, input format, feature order, or output contract.

<a id="paper"></a>

## 📄 Paper

**TxShield: Transaction-level RAG-enabled DeFi Attack Detection** — Under review at **IEEE Transactions on Information Forensics and Security (TIFS)**. This work has not been accepted or published yet.

The manuscript is the authoritative reference for the algorithm, datasets, and reported evaluation. Add the final publication link, BibTeX entry, and `CITATION.cff` when the paper metadata is finalized.

## 🙏 Acknowledgments

Historical attack cases are derived from the [DeFiHackLabs](https://github.com/SunWeb3Sec/DeFiHackLabs) ecosystem. TxShield also builds on Jina Embeddings, Chroma, LangChain, XGBoost, Ollama, and Hugging Face Transformers.

---

<p align="center">
  <a href="https://github.com/oumo0/TxShield"><img src="https://img.shields.io/badge/⭐_Star_TxShield-4F46E5?style=for-the-badge&logo=github&logoColor=white" alt="Star TxShield"></a>
  <a href="https://github.com/oumo0/TxShield/issues"><img src="https://img.shields.io/badge/🐛_Report_Issue-EF4444?style=for-the-badge&logo=github&logoColor=white" alt="Report Issue"></a>
  <a href="https://github.com/oumo0/TxShield/fork"><img src="https://img.shields.io/badge/🍴_Fork_Repository-10B981?style=for-the-badge&logo=github&logoColor=white" alt="Fork Repository"></a>
  <br>
  <strong>Research software for studying history-guided DeFi attack detection.</strong>
  <br>
  <sub>Do not use TxShield as a substitute for professional smart-contract audits or production security controls.</sub>
</p>
