# BioHub Cell Tracking During Development

## Project Goal

This project aims to build a state-of-the-art AI pipeline for the Kaggle competition:

**BioHub - Cell Tracking During Development**

The objective is to automatically:

- Detect cells in 3D microscopy volumes.
- Track cells across time.
- Detect cell division events.
- Reconstruct complete lineage graphs.
- Generate a valid `submission.csv`.

---

# Competition Information

Competition Type:

3D + Time Cell Tracking

Input:

- Zarr Volumes
- GEFF Graph Files

Output:

submission.csv containing:

- Nodes
- Edges

---

# Long-Term Goal

Target:

Top Kaggle Ranking

The project is designed to be modular, reproducible, and research-oriented.

---

# Development Roadmap

Phase 0
Project Foundation

Phase 1
Dataset Pipeline

Phase 2
Cell Detection

Phase 3
Node Extraction

Phase 4
Tracking

Phase 5
Division Detection

Phase 6
Submission

---

# Technologies

Python

PyTorch

MONAI

Zarr

NetworkX

NumPy

OpenCV

Scikit-image

Kaggle

---

# Current Status

Production-ready training workflow added with:
- automatic dataset detection for Kaggle input
- resumable checkpoints
- structured logs and summary reports under outputs
- memory-safe DataLoader defaults
- AMP-compatible trainer setup