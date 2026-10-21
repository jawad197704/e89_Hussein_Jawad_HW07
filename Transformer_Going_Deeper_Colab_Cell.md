# Part 0b — Going deeper (optional)

*Read this when you start **Problem 2**. It refers to the numbered boxes `[1]`–`[13]` in the diagram above.*

---

## Training vs. translating (inference)

| | Training | Translating (inference) |
|---|---|---|
| Decoder input | The full correct target, shifted right: `<s> me gusta el fútbol` | Starts with only `<s>` |
| How it runs | All positions are predicted **at once, in parallel**. The causal mask **[10]** stops cheating. | **One token per call:** predict, append, call again, until `</s>` (the 5-call table above) |
| Loss | `nn.CrossEntropyLoss` against `me gusta el fútbol </s>` | none |

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
