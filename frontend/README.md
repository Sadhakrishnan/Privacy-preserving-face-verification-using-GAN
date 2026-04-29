# Privacy-Preserving Face Verification 🔐

This project implements a Hybrid GAN + Differential Privacy framework for secure face verification.

## Features
- Face detection using MTCNN
- Embedding extraction using InsightFace
- GAN-based embedding obfuscation
- Differential Privacy noise injection
- Cosine similarity verification

## Tech Stack
- Python
- PyTorch
- FastAPI
- OpenCV

## Run Backend
```bash
uvicorn backend.main:app --reload