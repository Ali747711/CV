# sinoLLM learning path

Status: done — six guided notebooks implemented and locally verified.

Read the notebooks in order. Select **Python (mini-LLM PyTorch)** as the kernel, then Run All. Each notebook runs independently using the shared `lab.py` in this folder. It works when opened from the repository root, `mini_llm`, or this folder. Required packages are already in the project's `.venv-llm`: torch, matplotlib, and ipykernel. The lessons use CPU and two threads for predictable, small experiments. They do not install packages, download data, call an external API, or modify your existing sinoLLM checkpoint.

| Notebook | You will learn | Evidence to inspect |
|---|---|---|
| [01 — Trace one prediction](01_trace_one_prediction.ipynb) | Tokens, tensors, embeddings, logits, softmax, sampling | Shape trace and probability chart |
| [02 — Understand learning](02_understand_learning.ipynb) | Loss, gradients, backpropagation, optimizer, training vs inference | A parameter update and a learning curve |
| [03 — Rebuild attention](03_rebuild_attention.ipynb) | Q/K/V, causal masks, heads, residuals, normalization | Worked attention calculation and no-future-leak check |
| [04 — Run reliable experiments](04_reliable_experiments.ipynb) | Splits, leakage, baselines, validation, test, overfitting | Controlled comparison with multiple seeds |
| [05 — Fine-tuning and LoRA](05_instruction_tuning_and_lora.ipynb) | Pretraining, chat templates, response-only loss, frozen weights, low-rank adapters | Actual local pretraining, full fine-tuning, and LoRA updates |
| [06 — Build an AI system](06_build_an_ai_system.ipynb) | Retrieval, tools, serving, latency, reliability | Retrieval evaluation, a local HTTP request, input rejection and shutdown |

Every lesson includes definitions, a worked example, a visual, exercises, and expandable suggested answers. [GLOSSARY.md](GLOSSARY.md) is a reference, not a prerequisite. Executed outputs are saved so you can review without rerunning.

The models and data are intentionally tiny. Lesson 5 teaches genuine optimization mechanics on a synthetic task, not effective adaptation of a capable open model. Lesson 6 separates a reliable lookup/tool baseline from an unreliable tiny-model draft. It is a local teaching service, not a production deployment. Read the limitations beside each result.

## Task record

Request: turn the proposed six-stage learning path into reviewable learning materials with definitions.

Scope: six new guided notebooks, a shared implementation, a glossary, verification tools and this guide; add a navigation link to the root README. Preserve existing notebooks, model checkpoints, datasets, virtual environments, and other sessions' files. No deployment, external service or dependency installation. GitHub delivery was subsequently requested by the project owner.

Plan: define a common tiny model and reproducible data; build all six independent lessons; execute each in a fresh kernel; check behavior and boundary cases; inspect generated charts; record evidence here.

Definition of Done:

- [x] All six stages contain definitions, working examples, visuals, exercises and answers.
- [x] Each notebook validates and executes in a fresh kernel with no error output.
- [x] Behavioral checks cover causal masking, learning, held-out data separation, LoRA frozen weights, service input handling and clean shutdown.
- [x] Inspect rendered visual outputs and record exact verification results.

Verification on 2026-10-24 KST, current local working tree based on `11d8368`, Python 3.12.14 / PyTorch 2.14.1 / CPU:

```bash
.venv-llm/bin/python mini_llm/learning_path/verify_lessons.py
.venv-llm/bin/python -m unittest discover -s mini_llm/learning_path -p test_lab.py -v
```

- All six notebooks executed independently in fresh Jupyter kernels: **32 code cells**, no error outputs. Each complete notebook took about 9–11 seconds including kernel startup and plotting. Saved outputs are the execution evidence.
- After a diagram-spacing adjustment, reran lesson 1 with `verify_lessons.py 01_trace_one_prediction.ipynb`; it passed. After strengthening large-integer input rejection, reran all **11 behavioral tests**; all passed.
- Validated notebook schemas, parsed all code cells and shared Python files, checked local notebook links and `git diff --check`; all passed.
- Inspected rendered examples from every stage: shape-flow diagram, gradient update, attention matrices, experiment comparison, response mask, fine-tuning curves and application flow. Corrected arrows touching the embedding box in lesson 1.
- Lesson 2's transformer validation loss fell after training. Lesson 3's correct mask preserved earlier logits while the deliberately unmasked model leaked future information.
- Lesson 4 selected width 48 using mean validation loss 1.448 versus 1.800 for width 24, across two seeds with equal token budgets. Final selected-width mean test loss was 1.529. This is a tiny synthetic benchmark, not a general model ranking.
- Lesson 5 updated 6,724 adapter parameters versus 68,353 full-model parameters. Both adaptations generated **8/8 training words and 0/2 held-out words** exactly. These observed failures are retained and explained; fitting training words did not demonstrate general instruction following. Frozen-parameter checks passed.
- Lesson 6 retrieved the expected source on all three designed matching queries, returned no match for unrelated vocabulary, and showed a failed generated draft. Real loopback HTTP requests verified lookup/tool responses, invalid-request rejection and server shutdown. The service does not remain running.

Jupyter and HTTP checks required local-socket access outside the restricted sandbox. No package install, model download, global kernel registration or deployment was performed. Existing model artifacts and notebooks were preserved. No CUDA/MPS execution, VS Code interactive rendering check, broad retrieval benchmark, capable-model adaptation or production readiness is claimed.

The behavioral checks are reproducible from `test_lab.py`; `verify_lessons.py` supports optional notebook filenames to rerun only changed lessons. The shared implementation is readable in `lab.py`, and the temporary loopback server is in `local_service.py`.

Next action: open lesson 1, review its definitions and saved outputs, then try its first exercise before moving on.
