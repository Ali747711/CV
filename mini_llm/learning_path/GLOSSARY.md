# sinoLLM glossary

Read each term beside its runnable example. These definitions describe the course's tiny character-level models; they do not imply human-like understanding.

## Data and shapes

| Term | Plain definition | Example or location |
|---|---|---|
| Token | Unit read or predicted by a language model | One character here; larger models often use subwords |
| Tokenizer | Mapping between text and token IDs | `lab.encode` and `lab.decode` |
| Vocabulary | Set of supported tokens | `lab.CHARS`, 97 entries |
| Tensor | An array with dimensions and a numeric type | Token IDs use integer tensors |
| Shape | Length of each tensor dimension | `(batch, time, vocabulary)` for logits |
| Batch | Several independent examples processed together | Eight windows per training update in the shared helper |
| Context window | Maximum input length for one forward pass | 48 characters by default |
| Target / label | Desired output used to calculate loss | Each input character's following character |
| Padding | Extra positions used to equalize example lengths | Right-padding in lesson 5 |
| EOS | Explicit end marker | Reserved `~` character in the toy instruction task |

## Model internals

| Term | Plain definition | Example or location |
|---|---|---|
| Parameter | A trainable stored number | Weight matrices, biases, embedding entries |
| Embedding | Trainable vector associated with an ID | `token_embedding` |
| Position embedding | Trainable vector associated with position in a window | `position_embedding` |
| Linear layer | Matrix multiplication plus optional bias | `nn.Linear` |
| Logit | Unnormalized output score | One per vocabulary entry per position |
| Softmax | Converts scores into probabilities summing to one | Along the vocabulary or attention-key dimension |
| Query / key / value | Three learned projections used to match and mix token information | `CausalSelfAttention` |
| Attention | Weighted combination of value vectors, with weights determined by query/key matches | Lesson 3's worked matrices |
| Causal mask | Blocks future positions from influencing earlier predictions | Lower-triangular allowed positions |
| Head | One attention projection subspace | Three heads of 16 features at width 48 |
| Residual connection | Adds a layer's input to its output | `x + layer(x)` |
| LayerNorm | Normalizes features within a position with learned scale/offset | `Block.norm1` and `norm2` |
| Feed-forward network | Nonlinear transformation applied separately at each position | `Block.feed_forward` |
| Transformer block | Attention and feed-forward transformations with normalization and residual paths | `Block` |

## Learning and adaptation

| Term | Plain definition | Example or location |
|---|---|---|
| Loss | Numeric penalty used to fit parameters | Cross-entropy here |
| Cross-entropy | For one true token, negative log of its predicted probability | Averaged over active target positions |
| Gradient | Local derivative of loss with respect to a parameter | `.grad` after `backward()` |
| Backpropagation | Computes gradients through a chain of operations | `loss.backward()` |
| Optimizer | Algorithm updating parameters from gradients | SGD in the small worked example; AdamW in the transformer |
| Learning rate | Setting controlling update size | `lr` |
| Step | One parameter update using a batch | `optimizer.step()` |
| Epoch | One complete pass through a defined dataset | Different from randomly sampled batch steps |
| Training mode | Layer behavior selected by `model.train()` | Enables dropout where configured; does not itself update weights |
| Evaluation mode | Layer behavior selected by `model.eval()` | Disables dropout; does not itself disable gradients |
| Inference | Computing predictions without training updates | `generate` uses `no_grad()` |
| Pretraining | Next-token training on text before task adaptation | Lesson 5's synthetic base model |
| SFT | Supervised fine-tuning using desired input/response examples | Toy `say <word>` task |
| Chat template | Exact format used to mark roles/turns | `User: ...\nAssistant: ...` here |
| Response-only loss | Only response targets contribute to optimization | Prompt/padding targets become `-100` |
| Teacher forcing | Training conditions later response predictions on the correct earlier response tokens | SFT loss computation |
| Frozen parameter | Parameter not updated by training | Base weights in the LoRA model |
| LoRA | Learn low-rank matrix factors that modify a frozen weight's computation | `LoRALinear` |
| Rank | Intermediate width controlling adapter capacity | `rank=4` |
| Checkpoint | Saved state needed to reconstruct a model or resume an experiment | These lessons do not write over existing checkpoints |

## Generation and evaluation

| Term | Plain definition | Example or location |
|---|---|---|
| Autoregressive generation | Repeatedly predict and append the next token | `SinoLM.generate` |
| Greedy decoding | Choose the highest-scoring next token | `greedy=True` |
| Sampling | Randomly select according to a probability distribution | `torch.multinomial` |
| Temperature | Rescales scores before softmax | Lower is more concentrated; no extra knowledge is added |
| Top-k | Restrict candidates to the k highest-scoring tokens | `top_k=12` |
| Seed | Initial pseudorandom state | Matching seeds help comparisons, but do not remove all backend differences |
| Baseline | Reference method against which changes are judged | Uniform-character guessing or fixed lookup |
| Hyperparameter | Setting chosen outside gradient descent | Width, learning rate, context length |
| Validation | Held-out data used for model/settings selection | Lesson 4 |
| Test | Data reserved for evaluation after model choices | Repeated tuning on it compromises its role |
| Leakage | Evaluation information improperly influences training/selection | Split sentences before overlapping windows |
| Overfitting | Better training fit without comparable held-out improvement | Inspect train and validation separately |
| Perplexity | Exponential of mean next-token cross-entropy | Tokenization and evaluation data must be comparable |
| Exact match | Fraction of answers identical to expected strings | Lesson 5 held-out words |

## Applications

| Term | Plain definition | Example or location |
|---|---|---|
| Retrieval | Select relevant information from a collection | Word-count cosine baseline in lesson 6 |
| Semantic embedding | Representation trained to capture useful meaning relationships | Not used by this lexical retrieval demo |
| RAG | Supply retrieved context as input before model generation | Tiny draft experiment in lesson 6 |
| Grounding | Support answer claims with evidence | A source ID alone does not prove support |
| Tool call | Structured invocation of an allowed external function | Explicit `sum` request here |
| Abstention | Return no answer when evidence is inadequate | `no_match`; lexical threshold alone is limited |
| Serving | Expose behavior through an interface | Local `POST /ask` |
| Latency | Time elapsed for a request | Local measurements, not production percentiles |
| Prompt injection | Untrusted content tries to redirect model instructions | Treat retrieved text as data, not permissions |
