<h1 align="center">🔐 TxShield: Transaction-level RAG-enabled DeFi Attack Detection</h1>

<p align="center">
  <em>Leveraging Historical Attack Knowledge for DeFi Security</em>
  <br>
  <a href="https://github.com/oumo0/TxShield">GitHub</a>
  ·
  <a href="#-quick-start">Quick Start</a>
  ·
  <a href="#-features">Features</a>
  ·
  <a href="#-paper">Paper</a>
</p>

<p align="center">
  <a href="https://github.com/oumo0/TxShield/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  </a>
  <a href="https://github.com/oumo0/TxShield/issues">
    <img src="https://img.shields.io/github/issues/oumo0/TxShield" alt="GitHub Issues">
  </a>
  <a href="https://github.com/oumo0/TxShield">
    <img src="https://img.shields.io/badge/Paper-ISSTA'26_Submission-important" alt="Paper: ISSTA '26">
  </a>
  <a href="https://github.com/oumo0/TxShield/stargazers">
    <img src="https://img.shields.io/github/stars/oumo0/TxShield?style=social" alt="GitHub Stars">
  </a>
</p>
<p align="center">
  <b>F1: 0.824 | Recall: 95.60% | Latency: 11.82s | Cost: $0.26/tx</b>
  <br>
  <i>Outperforms TrxGNNBERT by 7.6% and runs 20× faster</i>
</p>

---

## 🎯 What Makes TxShield Different?

While traditional tools rely on rigid rules, TxShield learns from historical attack patterns to detect sophisticated DeFi exploits.

|                  | **TxShield**                            | 🤖 **Traditional Tools**       |
| ---------------- | --------------------------------------- | ----------------------------- |
| **Approach**     | Semantic pattern matching               | Rule-based detection          |
| **Knowledge**    | Learns from 210+ historical attacks     | Static vulnerability patterns |
| **Adaptability** | Detects novel variants of known attacks | Misses custom pricing logic   |
| **Speed**        | 11.82s per transaction                  | ~5 minutes per transaction    |
| **Cost**         | $0.26 per transaction (Llama)           | High computational overhead   |

**Core Innovation**: TxShield understands *attack behavior*, not just code vulnerabilities. It asks: **"Have we seen this story before?"**

## ✨ Key Features

- **🧠 Intelligent Knowledge Base**: 210+ historical DeFi attacks with LLM-generated behavioral summaries
- **🔍 Two-Stage Retrieval**: Semantic search with confidence filtering (τ=0.85)
- **🎭 Adaptive Reasoning**: Switches between retrieval-augmented and zero-shot analysis
- **⚡ Real-time Detection**: ~15 seconds per transaction (20x faster than traditional tools)
- **📊 Explainable Decisions**: Confidence scores with evidence from similar historical cases

## 📊 Performance Highlights

### 🏆 Benchmark Results

| Dataset          | Metric   | TxShield   | Best Baseline      | Improvement |
| ---------------- | -------- | ---------- | ------------------ | ----------- |
| DeFiAttackBench  | F1-Score | **0.824**  | 0.766 (TrxGNNBERT) | +7.6%       |
| PriceAttackBench | Recall   | **95.60%** | 80.22% (DeFiScope) | +15.4%      |
| All Attacks      | Latency  | **15.02s** | ~5min (Others)     | 20x faster  |



### 🎯 Attack Type Performance

| Attack Type                          | Detection Rate | Difficulty    |
| ------------------------------------ | -------------- | ------------- |
| Reentrancy                           | 96.2%          | 🟢 Low         |
| Price & Oracle Manipulation          | 93.2%          | 🟢 Low         |
| Flash-loan Price Manipulation        | 91.7%          | 🟢 Low         |
| Business Logic Flaws                 | 89.6%          | 🟡 Medium      |
| Liquidity & Migration Exploits       | 87.5%          | 🟡 Medium      |
| Arbitrary Call / External Call       | 86.7%          | 🟡 Medium      |
| Access Control                       | 79.3%          | 🟡 Medium      |
| Fake Market / MEV / Sandwich Attacks | 75.0%          | 🔴 High        |
| Signature & Validation Issues        | 57.1%          | 🔴 High        |
| Cryptographic Exploits               | <50%           | 🔴 Fundamental |


## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/oumo0/TxShield.git
cd TxShield
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

### Basic Usage

```python
from main import analyze_contract

# Analyze transaction file
result = analyze_contract(
    query="Please analyze this transaction to determine if it's an attack.",
    system_input_path="path/to/transaction.json"
)

# Or directly pass JSON string
result = analyze_contract(
    query="Analyze this transaction",
    system_input_text=json_data_string
)

# Result structure
print(f"Result: {result.get('answer')}")
print(f"RAG Enabled: {result.get('used_rag')}")
print(f"Similarity Metrics: {result.get('similarity_info')}")
```

The system's design embodies the **separation of *interface simplicity* from *implementation complexity***. Users only need to provide basic parameters (*query statements*, *transaction data*), while the underlying engine executes a **multi‑layer analytical pipeline**: **vector retrieval**, **pattern matching**, **multi‑model inference**, and **structured parsing**. The application of the ***facade pattern*** renders complex smart‑contract security analysis **transparent to researchers**, while maintaining the **modularity** and **extensibility** of the analytical engine. In short, behind a **simple user interface** lies a **complete analytical pipeline** encompassing **retrieval‑augmented generation**, **dynamic filtering**, and **intelligent compression**.


## 🏗️ Architecture

<p align="center">
  <img src="assets/flowchart.jpg" alt="TxShield Architecture" width="700">
</p>

**📚 Historical Knowledge Base Construction**
- Collects execution traces from 210+ historical DeFi attacks
- Generates semantic behavioral summaries via LLMs
- Builds vector embeddings for similarity-based retrieval

**🔍 RAG-based Decision Engine**
- Preprocesses incoming transactions
- Retrieves similar historical cases using semantic search with confidence filtering (τ=0.85)
- Performs adaptive reasoning based on match quality
- Produces structured, explainable decisions

## 📁 Project Structure

```
TxShield/
├── contract_analyzer/              
│   ├── analysis/                   # Performance evaluation and benchmarking
│   │   ├── __init__.py
│   │   ├── metrics.py             
│   │   ├── metrics_export.py
│   │   └── visualizer.py          
│   ├── api/                       
│   │   ├── __init__.py
│   │   └── analyzer_api.py        # FastAPI implementation
│   ├── config/                    management
│   │   ├── __init__.py
│   │   └── settings.py           
│   ├── core/                      
│   │   ├── __init__.py
│   │   ├── base_chain.py          # Abstract base for processing chains
│   │   ├── embeddings.py         
│   │   ├── message_manager.py     # LLM message orchestration
│   │   ├── middleware.py
│   │   ├── rag_chain.py           # Retrieval-Augmented Generation implementation
│   │   ├── router.py              # Routing between analysis modes
│   │   └── vector_store.py        # Vector database operations
│   ├── models/                    
│   │   ├── __init__.py
│   │   ├── base_model.py         
│   │   ├── deepseek_model.py      
│   │   ├── embedding_model.py    
│   │   ├── ernie_model.py        
│   │   ├── gitee_wrapper.py      
│   │   ├── llama_model.py        
│   │   └── local_model.py       
│   ├── utils/                     # Utility functions
│   │   ├── __init__.py
│   │   ├── compression_utils.py
│   │   ├── data_loader.py         # Dataset loading
│   │   ├── debug_utils.py
│   │   ├── file_utils.py
│   │   └── global_instances.py    # Global state management
│   ├── __init__.py
│   ├── global_instances.py
│   └── main.py
```


## 🛠️ Configuration

## Environment Variables

Create a `.env` file in your project root with the following credentials:

```bash
# API Keys (Required for cloud services)
BAIDU_API_KEY=your_baidu_api_key_here           # For Baidu ERNIE models
DEEPSEEK_API_KEY=your_deepseek_api_key_here     # For DeepSeek models
GITEE_API_KEY=your_gitee_api_key_here           # For Gitee AI models
EMBEDDING_API_KEY=your_embedding_api_key_here   # For embedding services

# Optional Local Deployment
LOCAL_API_URL=http://localhost:8000             # For local model servers
LOCAL_ENABLED=false                             # Set to "true" to enable local models

# Performance Tuning
SIMILARITY_THRESHOLD=SIMILARITY_THRESHOLD_SCORE                     # Similarity threshold (0-1, lower = stricter)
MAX_RETRIES=Maximum_API_retry_attempts                      
RETRY_DELAY=Delay
BATCH_SIZE=batch_size

# Network Configuration (Optional)
HTTP_PROXY=http://your-proxy:port               # HTTP proxy settings
HTTPS_PROXY=http://your-proxy:port              # HTTPS proxy settings
```

## Configuration Structure

The system uses a hierarchical configuration managed by `ConfigManager`:

### 1. Storage Configuration
```python
storage = StorageConfig(
    json_dir="contract_data_output",      # Directory for transaction JSON files
    vector_dir="vector_storage",          # Vector database storage
    results_dir="analysis_results",       # Analysis output directory
    cache_dir=".cache"                    # Cache directory
)
```

### 2. Model Configuration

The system supports multiple AI model providers with configurable settings:

```python
models = {
    "baidu": ModelConfig(
        provider="baidu",
        api_key="your_key",                # From environment variable
        api_url="url_to_baidu_service",
        max_retries=3,
        timeout=60,
        enabled=True
    ),
    "deepseek": ModelConfig(...),          # DeepSeek configuration
    "gitee": ModelConfig(...),             # Gitee AI configuration
    "local": ModelConfig(...)              # Local model server configuration
}
```

### 3. Embedding Configuration
```python
embedding_model = "embedding-model-name"    # Embedding model identifier
embedding_api_key = "your_embedding_key"    # Embedding service API key
embedding_url = "url_to_embedding_service"  # Embedding service endpoint
```

## Deployment Options

### Cloud-Based Deployment (Recommended for Experiments)
- **Advantages**: No local GPU requirements, quick setup
- **Models**: Baidu ERNIE, DeepSeek, Gitee AI
- **Requirements**: Valid API keys for selected providers
- **Best for**: Proof-of-concept, testing, resource-constrained environments

### Local Deployment (Recommended for Production)
- **Advantages**: Full control, reduced latency, enhanced privacy
- **Requirements**: 
  - Local GPU with sufficient VRAM (≥ 8GB recommended)
  - Model serving infrastructure (Ollama, vLLM, etc.)
  - `LOCAL_ENABLED=true` in `.env`
- **Configuration**: 
  ```bash
  LOCAL_API_URL=http://localhost:8000
  LOCAL_ENABLED=true
  ```

### Hybrid Deployment
- Use cloud APIs for embedding services
- Use local models for inference
- Balance between cost, performance, and privacy

## Usage Examples

### 1. Basic Configuration
```python
from config import CONFIG

# Access configuration values
threshold = CONFIG.similarity_threshold
models = CONFIG.get_enabled_models()
storage_path = CONFIG.storage.vector_dir
```

### 2. Model Selection
```python
class ContractAnalyzer:
    def _initialize_components(self):
        # Current implementation (Baidu ERNIE):
        self.llm = ErnieLLMWrapper(BaiduErnieAI())
        
        # To switch to another model:
        # from .llama_model import Llama3InstructAI
        # self.llm = ErnieLLMWrapper(Llama3InstructAI())
```

## ⚠️ Limitations & Future Work

**Current Limitations:**
- Cryptographic exploits (<50% detection rate)
- Multi-step cross-contract attacks
- Truly novel attack mechanisms

**Future Directions:**
1. Formal verification integration
2. Cross-contract call-graph analysis
3. Automated knowledge base expansion
4. Multi-modal detection approaches

## 🤝 Contributing

We ❤️ contributions! Here's how you can help:

1. **Report Bugs**: Found an issue? [Open a ticket](https://github.com/oumo0/TxShield/issues)
2. **Suggest Features**: Have an idea? Start a discussion
3. **Add Attack Patterns**: Submit new historical attacks
4. **Improve Documentation**: Fix typos, add examples
5. **Write Tests**: Help us improve coverage


## 📜 License

TxShield is released under the **MIT License**. See [LICENSE](LICENSE) for details.

> **Note**: This is research software. Use in production at your own risk. Always conduct security audits.

## 📄 Paper

Submitted to **ISSTA 2026**. Citation details will be available upon publication.

## 🙏 Acknowledgments

- Historical attack data from [DeFiHackLabs](https://github.com/SunWeb3Sec/DeFiHackLabs)
- The open-source DeFi security community

---

<p align="center">
  <b>Join the mission to secure decentralized finance</b>
  <br>
  <i>Because every transaction deserves a guardian.</i>
</p>

<p align="center">
  <a href="https://github.com/oumo0/TxShield">
    <img src="https://img.shields.io/badge/⭐_Star_TxShield-4F46E5?style=for-the-badge&logo=github&logoColor=white" alt="Star TxShield">
  </a>
  <a href="https://github.com/oumo0/TxShield/issues">
    <img src="https://img.shields.io/badge/🐛_Report_Issue-EF4444?style=for-the-badge&logo=github&logoColor=white" alt="Report Issue">
  </a>
  <a href="https://github.com/oumo0/TxShield/fork">
    <img src="https://img.shields.io/badge/🍴_Fork_Repository-10B981?style=for-the-badge&logo=github&logoColor=white" alt="Fork Repository">
  </a>
</p>

<p align="center">
  Built with ❤️ by security researchers, for the DeFi community
</p>

---

<details>
<summary>📊 Additional Performance Metrics</summary>

### 🏆 Overall Performance (DeFiAttackBench: 728 Transactions)

| Model                  | Accuracy  | Precision | Recall    | F1-Score  | Latency |
| ---------------------- | --------- | --------- | --------- | --------- | ------- |
| **GPT‑4.1‑mini**       | **0.812** | 0.774     | **0.882** | **0.824** | 15.02s  |
| Llama‑3.1‑8B‑Instruct  | 0.778     | 0.741     | 0.854     | 0.793     | 11.82s  |
| Baidu ERNIE‑Speed‑128k | 0.794     | 0.755     | 0.871     | 0.809     | 12.66s  |
| TrxGNNBERT (Baseline)  | 0.759     | 0.760     | 0.773     | 0.766     | ~5min   |

</details>