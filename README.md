# Computer Vision & Deep Learning Lab

A hands-on learning repository exploring Computer Vision and Deep Learning with **OpenCV** and **PyTorch**, progressing from real-time webcam face tracking to building and training a **Decoder-Only Transformer (Mini-LLM / Baby GPT)** from scratch.

---

## 📁 Repository Structure

```text
.
├── computer_vision/
│   └── main.py                     # Real-time face detection using OpenCV & AVFoundation
├── pytorch_basics/
│   ├── PyTorch.ipynb               # PyTorch fundamentals, tensors & linear regression workflow
│   └── assets/                     # Architecture and concept diagrams
│       ├── machine-learning-explained.svg
│       └── neural-network.excalidraw
├── mini_llm/
│   ├── mini_gpt.ipynb              # Decoder-Only Transformer (Mini-LLM) built from scratch
│   ├── chat.py                     # Interactive streaming CLI to talk with the model
│   ├── input.txt                   # Tiny Shakespeare training corpus (~1.1 MB)
│   ├── mini_gpt_shakespeare.pth    # Trained PyTorch model checkpoint (211K parameters)
│   ├── sinoLLM.ipynb               # Alternative educational tiny transformer experiment
│   ├── sinoLLM_README.md           # Documentation for sinoLLM
│   └── sinoLLM_artifacts/          # Artifacts and checkpoint for sinoLLM
├── requirements.txt                # Unified Python dependencies
└── .gitignore                      # Git ignore rules for venvs, caches, and system files
```

---

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Ali747711/CV.git
cd CV
```

### 2. Create and Activate Virtual Environment
Using Python 3.10 – 3.12:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*(Note: On Apple Silicon Macs, PyTorch utilizes Metal Performance Shaders (`mps`) out of the box for fast GPU training!)*

---

## 🎯 Modules & Usage

### 1. Computer Vision: Real-time Face Detection
Tracks faces from the MacBook built-in camera using OpenCV Haar Cascade classifiers:
```bash
python computer_vision/main.py
```
* Uses macOS `AVFoundation` to avoid iPhone Continuity Camera index conflicts.
* Press `q` to quit the video stream.

---

### 2. PyTorch Basics & Linear Regression
Open [`pytorch_basics/PyTorch.ipynb`](pytorch_basics/PyTorch.ipynb) in VS Code or Jupyter:
* Explores tensor operations, shapes, and GPU agnostic setup.
* Implements an end-to-end linear regression model with `nn.Module`, `L1Loss`, `SGD`, and Matplotlib visualization.

---

### 3. Mini-LLM (Baby GPT) from Scratch

For a guided course with definitions, visual explanations and exercises, start with the [six-stage sinoLLM learning path](mini_llm/learning_path/README.md). Each notebook runs independently and includes saved example outputs.

Open [`mini_llm/mini_gpt.ipynb`](mini_llm/mini_gpt.ipynb) in VS Code or Jupyter:
* Builds a full Decoder-Only Transformer architecture cell-by-cell:
  * Character-level Tokenizer (`encode` / `decode`)
  * Queries ($Q$), Keys ($K$), Values ($V$) with Causal Masking
  * Multi-Head Attention, Feed-Forward Networks, LayerNorm, and Residual connections
  * Next-token prediction loss (`CrossEntropyLoss`)
* Trains in ~1–2 minutes on laptop GPU (`mps` / `cuda`).

#### Interactive Terminal Chat:
Talk to the trained Shakespeare model live:
```bash
python mini_llm/chat.py
```
* Streams generated tokens in real time.
* Adjust creativity on the fly: type `/temp 0.3` for strict English, or `/temp 0.8` for wild poetry.
* Type `exit` or `quit` to leave.

---

## 📜 License
MIT License. Created for educational and learning purposes.
