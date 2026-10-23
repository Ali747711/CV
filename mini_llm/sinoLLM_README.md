# sinoLLM: learn to build a tiny language model

Open `sinoLLM.ipynb` in VS Code or Jupyter, select **Python (mini-LLM PyTorch)** as the kernel (the registered `venv-llm` kernel, using `.venv-llm/bin/python`), and run the cells in order. The saved outputs show an actual completed run. Restart the kernel and Run All to reproduce the whole experiment; avoid re-running the training cell alone when comparing fresh runs.

For a different environment, install `sinoLLM-requirements.txt` into its virtual environment and select that Python interpreter as the notebook kernel. No installation was needed or performed in the existing environment.

The notebook builds a character-level decoder-only transformer from random weights using PyTorch. It includes diagrams, a causal-mask visualization, training curves, next-character probabilities, learned attention, before/after text, and checkpoint saving/loading. All model implementation code is in the notebook. It uses 68,353 parameters, two blocks, three attention heads per block, and a 48-character context window.

The default data is 80 authored synthetic sentences. Whole sentences are shuffled with a fixed seed and split 85/15 before batching. Validation contains held-out subject/action combinations but shares the training grammar and vocabulary. This is a narrow learning demonstration, not a broad language benchmark. Set `DATA_PATH = Path("input.txt")` to experiment with the existing local corpus; the default run does not read or modify it. ASCII text is supported; other characters produce an explicit error.

Running the notebook writes `sinoLLM_artifacts/sinoLLM.pt`. It contains CPU model weights, architecture settings, tokenizer characters, and the configured training-step count. It is an inference checkpoint, not an exact training-resume checkpoint. No optimizer state is stored.

## Task record

Status: done — implemented and locally verified. No commit, push, or deployment.

Owner / branch / working directory: this sinoLLM task; `fix/my-change`; `/Users/mac/Desktop/LearnDEV/computer-vision/opencv`.

Request: create an educational small language model called sinoLLM in a Jupyter notebook.

Scope: new sinoLLM notebook, setup requirements, this guide, and the generated checkpoint. Existing Python files, other notebooks, datasets, diagrams, virtual environments, and Git configuration are outside this task's scope and were preserved. No external training service or paid API is used.

Plan completed: inspect environment; build an explained transformer notebook; execute every cell in a fresh kernel; check attention causality, loss, generation and checkpoint behavior; inspect rendered charts.

### Definition of Done and verification

Verified on 2026-10-24 KST with Python 3.12.14, PyTorch 2.14.1, matplotlib 3.11.2, ipykernel 7.4.0, nbclient 0.11.0, and nbformat 5.11.1. Full notebook execution used the Mac's MPS device; the additional boundary checks used CPU. CUDA was not tested. Runtime varies by hardware and backend.

- [x] Valid notebook with sequential explanations and visualizations — `nbformat.validate` passed; all 14 code cells executed through `nbclient.NotebookClient` in a fresh local kernel with no error outputs.
- [x] A model trained from random weights and produced text — 300 training steps; fixed-batch training loss 4.726 to 0.329, validation loss 4.750 to 0.456. Before/after samples and plots are saved in the notebook. Generated text remains imperfect.
- [x] No future-token leakage — attention rows summed to one, upper-triangle weights were zero, and changing future tokens left earlier logits unchanged.
- [x] Portable inference checkpoint — save/reload reproduced the same logits on MPS; loading and inference also passed on CPU.
- [x] Relevant boundaries — CPU checks passed for finite loss and gradients, invalid/empty/oversized contexts, invalid temperature/top-k/token-count options, generation beyond the context length, and restoration of training mode after generation.
- [x] Visual evidence — inspected the actual rendered learning curve and learned-attention heatmap extracted from notebook outputs.

Execution procedure: run every code cell in a fresh Jupyter kernel using `.venv-llm/bin/python`; inspect outputs and embedded assertions. Additional CPU verification instantiated the classes from the notebook and loaded the generated checkpoint, then exercised the boundaries listed above. Temporary verification scripts were kept outside the project.

The saved notebook is the durable evidence for the end-to-end execution and assertions. No changes were made to the installed packages or existing notebooks. Independent review was not required for this isolated educational addition; no production readiness is claimed.

Next action: open the notebook, read top to bottom, then try the one-variable-at-a-time experiments in section 13.

Kernel picker correction (2026-10-24): confirmed the existing `venv-llm` registration points to the project interpreter. Updated the notebook kernelspec to match its registered name and display label. No global kernel configuration or model code changed. Reopen the notebook to refresh its requested kernel.

## Talking to the trained model

Run the last cell in section 14. Type in the VS Code/Jupyter input prompt and press Enter. Type `quit` to finish. If the kernel has restarted, run the earlier sections first or use Run All. Each prompt is independent; output is a model-generated text continuation, not an instruction-following answer. Messages are not used for training. No additional dependencies or external service are used.

Chat addition (2026-10-24): added an interactive notebook loop and continuation function. Existing model architecture and training cells were preserved.

Chat verification: `.venv-llm/bin/python /private/tmp/check_sino_chat.py` passed on 2026-10-24. Used the actual saved checkpoint on CPU and mocked only terminal input to exercise generated replies, long prompts, blank/unsupported input recovery, quit, interrupt/EOF, and missing-model guidance. Model weights remained unchanged. The notebook validates; the VS Code input-box interaction itself has not been manually exercised. The new interactive cell intentionally waits for user input during Run All.
