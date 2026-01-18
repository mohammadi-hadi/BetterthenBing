<div align="center">

# BetterThanBing

### Neural Information Retrieval with Semantic Search

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.0+-red.svg)](https://pytorch.org/)

*Exploring dense retrieval methods that outperform traditional keyword-based search*

---

</div>

## Overview

This project implements semantic search using dense vector representations, demonstrating how neural information retrieval can achieve better results than traditional keyword-based search engines. It uses BEIR benchmarks and Pyserini for efficient retrieval.

## Features

- **Dense Retrieval**: Embedding-based document search using sentence transformers
- **BEIR Integration**: Benchmark evaluation on standard IR datasets
- **FAISS Indexing**: Efficient similarity search with HNSW index
- **Pyserini**: Industry-standard search toolkit integration

## Technology Stack

- **BEIR**: Heterogeneous benchmark for information retrieval
- **Pyserini**: Python interface to Lucene and FAISS
- **Sentence Transformers**: Neural text embeddings
- **FAISS**: Facebook AI similarity search

## Quick Start

```bash
# Clone the repository
git clone https://github.com/mohammadi-hadi/BetterthenBing.git
cd BetterthenBing

# Install dependencies
pip install beir pyserini sentence-transformers faiss-gpu

# Open the notebook
jupyter notebook XAI_Project.ipynb
```

## Pipeline

### 1. Data Preparation
Download and process BEIR benchmark datasets (e.g., FiQA financial Q&A).

### 2. Document Encoding
Generate dense embeddings using sentence transformers:
```python
# Using XtremeDistil model for efficient encoding
encoder: microsoft/xtremedistil-l6-h256-uncased
```

### 3. Index Building
Create FAISS HNSW index for fast approximate nearest neighbor search.

### 4. Semantic Search
Query the index with natural language questions and retrieve relevant documents.

## Example

```python
from pyserini.search.faiss import AutoQueryEncoder, FaissSearcher

encoder = AutoQueryEncoder('microsoft/xtremedistil-l6-h256-uncased')
searcher = FaissSearcher(index_dir="fiqa_index", query_encoder=encoder)

hits = searcher.search('what is a lobster roll')
for i, hit in enumerate(hits[:10]):
    print(f'{i+1} {hit.docid} {hit.score:.5f}')
```

## Repository Structure

```
BetterthenBing/
├── XAI_Project.ipynb  # Main notebook with implementation
├── LICENSE            # MIT License
├── CONTRIBUTING.md    # Contribution guidelines
└── README.md         # This file
```

## Requirements

- Python 3.8+
- beir
- pyserini
- sentence-transformers
- faiss-gpu (or faiss-cpu)
- tensorflow_text

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaboration inquiries:
- **Hadi Mohammadi** - Utrecht University
- **Email**: [h.mohammadi@uu.nl](mailto:h.mohammadi@uu.nl)
- **Website**: [mohammadi.cv](https://mohammadi.cv)
