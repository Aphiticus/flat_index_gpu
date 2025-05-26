# flat_index_gpu

A fast, simple brute-force vector search on GPU or CPU  
—an easy, plug-and-play alternative to FAISS specifically for fast vector searching, not associative lookup for Windows and Linux.

Just drop `flat_index_gpu.py` into your project and get started.

---
When to Use
You need fast, accurate nearest neighbor search for dense vectors.

You want a simple, dependency-free alternative to FAISS, especially on Windows.

Your dataset fits in GPU or CPU memory (for huge datasets, use approximate/partitioned methods).
---

## Features

- Fast brute-force nearest neighbor search using PyTorch (GPU or CPU)
- Minimal API: add vectors, search, save, load
- Works out-of-the-box on Windows or Linux
- Supports `float16` and `float32` automatically
- Only requires [numpy](https://numpy.org/) and [PyTorch](https://pytorch.org/)

---

## Installation

1. Download or copy `flat_index_gpu.py` into your project folder.

2. Install the dependencies if needed:

pip install torch numpy

yaml
Copy
Edit

---

## Quick Start

**Step 1: Import and create the index**

```python
import numpy as np
from flat_index_gpu import FlatIndexGPU

# Create random database vectors and query vectors
db = np.random.rand(1000, 32).astype(np.float32)
queries = np.random.rand(5, 32).astype(np.float32)

index = FlatIndexGPU(dim=32)      # Automatically uses GPU if available
Step 2: Add your vectors

python
Copy
Edit
index.add(db)
Step 3: Search for nearest neighbors

python
Copy
Edit
ids, dists = index.search(queries, topk=3)

print("Indices:\n", ids)
print("Distances:\n", dists)
How to Use in Your Script
Just follow these three steps in your code:

from flat_index_gpu import FlatIndexGPU

index = FlatIndexGPU(dim=your_vector_dim)

index.add(your_database_vectors)

ids, dists = index.search(your_query_vectors, topk=k)

Example: Save and Load the Index
python
Copy
Edit
index.save("my_index.pt")

# Later or in another script
index2 = FlatIndexGPU(dim=32)
index2.load("my_index.pt")
ids2, dists2 = index2.search(queries, topk=3)

License
MIT License



