# Assignment 07 — Hussein Jawad
**E89 Deep Learning · NLP with Transformers** (based on Géron, *Hands-On ML*, Ch. 15)

---

# Part 0 — What is a Transformer? A visual primer

## The idea in 30 seconds

A **Transformer** is a neural network that turns one sequence of **tokens** (words or pieces of words) into another. Translation is the classic example. It was introduced in the 2017 paper *"Attention Is All You Need"*.

**Why "Transformer"?** Apart from the embedding and output layers, **every layer takes a tensor of shape `[batch size, sentence length, embedding size]` and returns a tensor of exactly the same shape**. The tensor stays the same size. What changes is what each token's vector *means*: it is **gradually transformed**, layer by layer, until it captures what the token means *in this particular sentence*.

> **Example (Géron):** the word **"like"** is vague on its own ("I'm *like* a cat" vs. "I *like* my cat"). After the encoder has seen the whole sentence "I like soccer", the vector for "like" captures the meaning *"to be fond of"* and the fact that it is a verb.

The model has two halves:

| Half | Job | Analogy |
|---|---|---|
| **Encoder** (left of the diagram) | Reads the whole source sentence and outputs **one context-rich vector per token** | A translator who reads the full sentence and takes detailed notes |
| **Decoder** (right of the diagram) | Takes the encoder's notes plus **the translation so far**, and predicts the **next token** | A writer who writes the translation one word at a time, checking the notes before each word |

## The running example (from Géron)

We translate **"I like soccer" → "me gusta el fútbol"**. The decoder is called **once per output token**, and each call adds one token:

| Call | Decoder input (translation so far) | Predicted next token |
|---|---|---|
| 1 | `<s>` | `me` |
| 2 | `<s> me` | `gusta` |
| 3 | `<s> me gusta` | `el` |
| 4 | `<s> me gusta el` | `fútbol`  ← *shown in the diagram below* |
| 5 | `<s> me gusta el fútbol` | `</s>` → **stop** |

`<s>` = start-of-sequence token, `</s>` = end-of-sequence token.

---

## The architecture at a glance

Read the diagram **bottom → top**. The numbers `[1]`–`[13]` match the table that follows.

```text
                                              OUTPUT PROBABILITIES (next token)
                                           P("fútbol")=0.92  P("partido")=0.03
                                                           ▲
                                               ┌───────────────────────┐
                                               │ [13] Softmax  (*)     │
                                               └───────────────────────┘
                                                           ▲
                                               ┌───────────────────────┐
                                               │ [12] Linear → logits  │
                                               └───────────────────────┘
                                                           ▲
                                          ┌──────────────────────────────────┐
                                          │ DECODER LAYER          [7] × N   │
                                          │   ┌─────────────────────────┐    │
                                          │   │ [5] Add & Norm          │    │
                                          │   │ [6] Feed Forward        │    │
                                          │   └─────────────────────────┘    │
                                          │                ▲                 │
                                          │   ┌─────────────────────────┐    │
                   encoder output (K, V)  │   │ [5] Add & Norm          │    │
                 ┌────────────────────────┼──►│ [11] Multi-Head         │    │
                 │ fed to EVERY           │   │      Cross-Attention    │    │
                 │ decoder layer          │   └─────────────────────────┘    │
                 │                        │                ▲                 │
 ┌───────────────┴──────────────┐         │   ┌─────────────────────────┐    │
 │ ENCODER LAYER      [7] × N   │         │   │ [5] Add & Norm          │    │
 │  ┌────────────────────────┐  │         │   │ [10] Masked Multi-Head  │    │
 │  │ [5] Add & Norm         │  │         │   │      Self-Attention     │    │
 │  │ [6] Feed Forward       │  │         │   └─────────────────────────┘    │
 │  └────────────────────────┘  │         └────────────────▲─────────────────┘
 │               ▲              │                          │
 │  ┌────────────────────────┐  │                          │
 │  │ [5] Add & Norm         │  │                          │
 │  │ [4] Multi-Head         │  │                          │
 │  │     Self-Attention     │  │                          │
 │  └────────────────────────┘  │                          │
 └───────────────▲──────────────┘                          │
                 │                                         │
                (+)◄── [3] Positional                     (+)◄── [3] Positional
                 ▲     Embedding                           ▲     Embedding
     ┌───────────────────────┐                 ┌───────────────────────┐
     │ [2] Input Embedding   │                 │ [9] Output Embedding  │
     └───────────────────────┘                 └───────────────────────┘
                 ▲                                         ▲
                 │                                         │
                 │                                         │
            [1] Inputs                        [8] Outputs (shifted right)
          "I like soccer"                          "<s> me gusta el"
```

- **N** = number of stacked layers: **6 in the paper**, **2 in Géron's small model**.
- **(\*) Softmax:** in PyTorch this is *not* a layer inside the model. The model outputs **logits**, and `nn.CrossEntropyLoss` applies the softmax internally. Call `F.softmax()` if you want the probabilities.

---

## Each element, step by step

**Shapes:** `B` = batch size, `L_src` / `L_tgt` = longest source / target sentence in the batch, `d` = embedding size (`d_model`: **512** in the paper, **128** in Géron's small model), `V` = vocabulary size.

| # | Element | What it does (plain English) | Output shape |
|---|---|---|---|
| **1** | **Inputs** | The source sentence as **token IDs** (integers). Shorter sentences are padded with `<pad>` so every sentence in the batch has the same length. | `[B, L_src]` |
| **2** | **Input Embedding** | `nn.Embedding` looks up a **vector of `d` numbers** for each token ID. At this point "like" still has its vague, out-of-context meaning. | `[B, L_src, d]` |
| **3** | **Positional Embedding** `(+)` | Attention is **position-agnostic**: without help, "dog bites man" and "man bites dog" look the same. So **position vector #i is added to token #i**. The paper used fixed sine/cosine waves. Géron uses **learned** position vectors (`nn.Parameter`) plus dropout, which works just as well. | same |
| **4** | **Multi-Head Self-Attention** (encoder) | Each token **looks at every token in the same sentence**, including itself, and pulls in what is relevant. This is where "like" notices the subject "I" and works out that it is a verb. "Multi-head" means this happens in **several heads in parallel** (8 in the paper, 4 in Géron), and each head can focus on a different feature (tense, meaning, …). | same |
| **5** | **Add & Norm** | **Add:** a skip connection adds the sub-layer's input back to its output. **Norm:** `LayerNorm` keeps the values in a stable range. Together they make deep stacks trainable. This wraps **every** sub-layer. In code: `norm1(src + dropout(attn))`. | same |
| **6** | **Feed Forward** | Two `Linear` layers with a ReLU in between: `d → 2048 → d`. It works on **each token separately**. Attention *mixes* information *between* tokens; the feed-forward block *processes* it *within* each token. | same |
| **7** | **N× stacking** | The layer (4 → 5 → 6 → 5) is repeated **N times**, and each layer refines the vectors further. The encoder's final output (PyTorch calls it the **memory**) is sent to **every** decoder layer. | `[B, L_src, d]` |
| **8** | **Outputs (shifted right)** | During training the target is `me gusta el fútbol </s>`, and the decoder is fed the same sentence **shifted right by one**: `<s> me gusta el fútbol`. Position *t* then has to predict token *t + 1*. This is called **teacher forcing**. | `[B, L_tgt]` |
| **9** | **Output Embedding + Positional** | Same as **2** and **3**, but for the target tokens. In Géron's code these are literally the **same layers**, because both languages share one vocabulary. | `[B, L_tgt, d]` |
| **10** | **Masked (causal) Self-Attention** (decoder) | Like **4**, but each token may only look at **itself and earlier tokens**. For example, "gusta" sees `<s>`, "me" and "gusta", but **not** "el" or "fútbol". Without the mask the model could copy the answer during training. | same |
| **11** | **Cross-Attention** (encoder → decoder) | The two halves **connect** here. **Queries (Q)** come from the decoder, and **keys (K) and values (V)** are the **encoder's final outputs**. When the decoder turns "el" into "fútbol", it pays close attention to **"soccer"**. | same |
| **12** | **Linear** | Maps each vector to **one score (logit) per vocabulary token**. Ideally the correct next token gets a high logit and every other token a low one. | `[B, L_tgt, V]` |
| **13** | **Softmax** | Turns the logits into **probabilities that sum to 1**, e.g. `P("fútbol") = 0.92`. In PyTorch it lives inside `nn.CrossEntropyLoss`. | `[B, L_tgt, V]` |

---

## Zoom in: how attention works

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

Think of a **library search**:

- **Q (Query):** *"what am I looking for?"*, asked by the token being updated.
- **K (Key):** *"what do I contain?"*, a label on every token.
- **V (Value):** *"here is my actual information"*.

It works in five steps:

1. **Score:** `QKᵀ` gives a similarity score for **every query/key pair**, an `[L_q, L_k]` grid. The grid grows with the **square** of the sentence length, which is why long inputs are expensive (relevant to **Problem 3**).
2. **Scale:** divide by `√d_k` so the softmax doesn't saturate. A saturated softmax gives tiny gradients and slow training.
3. **Mask:** set every forbidden score to `−∞`, so its weight becomes exactly 0. This is how padding and future tokens are hidden.
4. **Softmax** each row, so each token's weights add up to 1.
5. **Mix:** the output for each token is the **weighted sum of the values**.

**Toy example (illustrative numbers):** updating the token **"like"** in "I like soccer":

| Looks at → | `I` | `like` | `soccer` |
|---|---|---|---|
| attention weight | **0.50** | 0.30 | 0.20 |

New "like" = 0.50·V(`I`) + 0.30·V(`like`) + 0.20·V(`soccer`). It now "knows" it has the subject "I", so it is a verb.

**The two masks.** A **padding mask** ignores `<pad>` tokens (`*_key_padding_mask`). A **causal mask** hides future tokens in the decoder (`tgt_mask`). For decoder input `<s> me gusta el`, the causal mask looks like this (✓ = may look at):

| token ↓ looks at → | `<s>` | `me` | `gusta` | `el` |
|---|---|---|---|---|
| `<s>` | ✓ | | | |
| `me` | ✓ | ✓ | | |
| `gusta` | ✓ | ✓ | ✓ | |
| `el` | ✓ | ✓ | ✓ | ✓ |

Géron builds it with `torch.triu(full_mask, diagonal=1)`, which is `True` above the diagonal, meaning "hide".

**Multi-head attention** runs several small attentions side by side, so each head can focus on a different feature:

```text
                         ┌─► head 1: attention ─┐
token vectors ─► Linear ─┼─► head 2: attention ─┼─► concatenate ─► Linear ─► output
  (Q, K, V)    (per head)└─► … head h …        ─┘
```

---

## The diagram ↔ Géron's PyTorch code (`NmtTransformer`)

| Diagram box | Where it lives in the code |
|---|---|
| **[2]**, **[9]** embeddings | `self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)`, shared by source and target |
| **[3]** positional | `PositionalEmbedding`: `X + self.pos_embed[:X.size(1)]`, then dropout |
| **[4]–[11]** encoder + decoder stacks | `nn.Transformer(embed_dim, num_heads, num_encoder_layers=N, num_decoder_layers=N, batch_first=True)` |
| **[5]** Add & Norm | `Z = self.norm1(src + self.dropout(attn))` |
| **[6]** Feed Forward | `self.linear2(self.dropout(self.linear1(Z).relu()))` |
| **[10]** causal mask | `causal_mask = torch.triu(full_mask, diagonal=1)` → `tgt_mask=causal_mask, tgt_is_causal=True` |
| padding masks | `src_key_padding_mask`, `memory_key_padding_mask`, `tgt_key_padding_mask` |
| **[11]** cross-attention | `self.multihead_attn(Z, memory, memory)`: Q = decoder `Z`, K = V = encoder `memory` |
| **[12]** Linear | `self.output = nn.Linear(embed_dim, vocab_size)` |
| **[13]** Softmax | inside `nn.CrossEntropyLoss` |
| **Size** | paper: `embed_dim=512, num_heads=8, num_layers=6`; Géron's small model: `128, 4, 2` |

---

## Training vs. translating (inference)

| | Training | Translating (inference) |
|---|---|---|
| Decoder input | The full correct target, shifted right: `<s> me gusta el fútbol` | Starts with only `<s>` |
| How it runs | All positions are predicted **at once, in parallel**. The causal mask **[10]** stops cheating. | **One token per call:** predict, append, call again, until `</s>` (the 5-call table above) |
| Loss | `nn.CrossEntropyLoss` against `me gusta el fútbol </s>` | none |

---

## How this connects to HW07

| Problem | Transformer concept involved |
|---|---|
| **1 – spaCy:** `prince − man + woman` | **Embeddings** (box **[2]**): meaning stored as vectors, so vector arithmetic and **cosine similarity** capture relationships between words. |
| **2 – French → English** | Géron's `NmtTransformer` with the languages swapped: **French** goes into **[1]** and **English** comes out of **[13]**. Hold out 10% of the pairs as the test set. |
| **3 – GPU memory** | Memory holds the **model weights**, the **Adam optimizer state** (two extra values per weight), **activations**, and the **attention grids**, which grow with *L²*. Géron's cleanup recipe: `del` the objects, then `gc.collect()`, then `torch.cuda.empty_cache()`. |
| **4 – BLEU score** | Scores a translation by how many **n-grams** (1–4 word chunks) it shares with a human reference translation. Higher is better. |
