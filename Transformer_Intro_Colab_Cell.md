# What is a Transformer? — A visual primer for HW07

A **Transformer** is a neural network that turns one sequence into another, for example a French sentence into an English one. It was introduced in the 2017 paper *"Attention Is All You Need"* (Vaswani et al.). Older models (RNNs/LSTMs) read a sentence **one word at a time**. A Transformer reads the **whole sentence at once** and uses **attention** to work out which words matter to each other.

It has two halves:

- **Encoder (left):** *reads and understands* the source sentence (French).
- **Decoder (right):** *writes* the target sentence (English) **one word at a time**, looking back at the encoder's notes as it goes.

> **Analogy:** the encoder is a translator who reads the whole French sentence and takes detailed notes. The decoder writes the English version word by word. Before each word, it checks what it has written so far and the relevant parts of those notes.

---

## The architecture at a glance

Read the diagram **bottom → top**. The numbers `[1]`–`[13]` match the table below. Example: translating *"Je t'aime"* → *"I love you"*, with the model about to predict the word after "I".

```text
                                                 OUTPUT PROBABILITIES
                                             P("love")=0.91  P("like")=0.04
                                                           ▲
                                               ┌───────────────────────┐
                                               │ [13] Softmax          │
                                               └───────────────────────┘
                                                           ▲
                                               ┌───────────────────────┐
                                               │ [12] Linear           │
                                               └───────────────────────┘
                                                           ▲
                                          ┌──────────────────────────────────┐
                                          │ DECODER LAYER      [7] × N=6     │
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
 │ ENCODER LAYER  [7] × N=6     │         │   │ [5] Add & Norm          │    │
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
                 ▲     Encoding                            ▲     Encoding
     ┌───────────────────────┐                 ┌───────────────────────┐
     │ [2] Input Embedding   │                 │ [9] Output Embedding  │
     └───────────────────────┘                 └───────────────────────┘
                 ▲                                         ▲
                 │                                         │
                 │                                         │
            [1] Inputs                        [8] Outputs (shifted right)
            "Je t'aime"                             "<sos> I love"
```

---

## Each element, step by step

**Shape notation:** `B` = batch size, `L_in` / `L_out` = max input / output sentence length, `512` = model dimension `d_model`, `V` = vocabulary size.

| # | Element | What it does (plain English) | Output shape |
|---|---|---|---|
| **1** | **Inputs** | The source sentence as a list of **word IDs** (integers), e.g. `Je→17, t'→342, aime→905`. | `[B, L_in]` |
| **2** | **Input Embedding** | A lookup table that turns each word ID into a **512-number vector** that captures meaning. Similar words get similar vectors. | `[B, L_in, 512]` |
| **3** | **Positional Encoding** `(+)` | Attention has **no built-in sense of word order** ("dog bites man" = "man bites dog"). So each position gets a unique pattern (sine/cosine waves in the paper, or learned vectors in some implementations). That pattern is **added** to the embedding. | same |
| **4** | **Multi-Head Self-Attention** (encoder) | Every word **looks at every other word** in the same sentence and decides how much each one matters to it. "Multi-head" means **8 attention heads run in parallel**, and each can learn a different relationship (grammar, meaning, reference…). | same |
| **5** | **Add & Norm** | **Add:** a *residual / skip connection* adds the sub-layer's input back to its output, so information and gradients flow easily. **Norm:** *layer normalization* keeps the numbers in a stable range. This wraps **every** sub-layer. | same |
| **6** | **Feed Forward** | A small 2-layer neural network (`512 → 2048 → 512`) applied to **each word separately**. Attention *gathers* information; this step *processes* it. | same |
| **7** | **N× (stacking)** | The whole layer (4 → 5 → 6 → 5) is **repeated N = 6 times**. Each layer refines the understanding a little more. The encoder's final output, `[B, L_in, 512]`, is **one context-rich vector per input word**. | `[B, L_in, 512]` |
| **8** | **Outputs (shifted right)** | During **training** the decoder is given the correct target sentence with a start token `<sos>` added at the front: `<sos> I love you`. At each position it must predict the **next** word ("teacher forcing"). | `[B, L_out]` |
| **9** | **Output Embedding + Positional Encoding** | Same as **2** and **3**, but for the target-language (English) words. | `[B, L_out, 512]` |
| **10** | **Masked Multi-Head Self-Attention** (decoder) | Like **4**, but a **mask hides future words**, so position *t* can only see words *before* it. Without the mask, the model could simply copy the answer during training. | same |
| **11** | **Multi-Head Cross-Attention** (encoder–decoder) | This is where the two halves **connect**. The decoder asks *"which French words matter for the English word I'm about to write?"* and attends over the **encoder's output**. The encoder's final output is fed into **every one** of the N decoder layers. | same |
| **12** | **Linear** | Projects each 512-dim vector to **one score per word in the vocabulary** (called *logits*). | `[B, L_out, V]` |
| **13** | **Softmax** | Turns the scores into **probabilities that sum to 1**. The most likely word is the prediction, e.g. `P("love") = 0.91`. | `[B, L_out, V]` |

---

## Attention in one formula

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

Think of a **library search**:

- **Q (Query):** *"what am I looking for?"*, asked by the current word.
- **K (Key):** *"what do I contain?"*, a label on every word.
- **V (Value):** *"here is my actual information"*.

`QKᵀ` scores how well each query matches each key. **Softmax** turns those scores into weights. The output is a **weighted mix of the values**. In **self-attention** (4, 10), Q, K and V all come from the same sentence. In **cross-attention** (11), **Q comes from the decoder** and **K, V come from the encoder**.

---

## Training vs. translating (inference)

| | Training | Inference (actual translation) |
|---|---|---|
| Decoder input | The full correct target, shifted right: `<sos> I love you` | Starts with only `<sos>` |
| How it runs | All positions are predicted **in parallel**. The mask (10) prevents cheating. | **One word at a time:** predict a word, append it, feed it back in, and repeat until `<eos>`. |
| Example | Predict `I love you <eos>` all at once | `<sos>` → "I" → "I love" → "I love you" → `<eos>` |

---

## How this connects to HW07

| Problem | Transformer concept involved |
|---|---|
| **1 – spaCy:** `prince − man + woman` | **Word embeddings** (element **2**): meaning is stored as vectors, so vector arithmetic and cosine similarity capture relationships between words. |
| **2 – French → English translation** | Training the **full encoder–decoder** above: French goes into **[1]**, and English comes out of **[13]**. |
| **3 – GPU memory** | Transformers are memory-hungry: attention compares **every word with every other word** (memory grows with `L²`), and there are 6 stacked layers of weights. |
| **4 – BLEU score** | Measures how closely the decoder's output **[13]** overlaps with human reference translations. |
