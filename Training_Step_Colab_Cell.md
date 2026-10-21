# Zoom in: one training step on "I like soccer"

**Setup:** the encoder has already made its 3 notes (**I**, **like**, **soccer**). The weights are still **random**. To keep it tiny, the Spanish vocabulary has only 5 words: `me`, `gusta`, `el`, `fútbol`, `</s>`.

## What the decoder gets, and what it must predict

During training we already know the answer, so the **whole correct sentence, shifted right**, goes in at once:

| Slot | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| **Input** | `<s>` | me | gusta | el | fútbol |
| **Must predict** | **me** | **gusta** | **el** | **fútbol** | **`</s>`** |

## Slot 1, step by step

| Step | What happens at slot 1 | Example (random weights) |
|---|---|---|
| **1. Embed** [9] | `<s>` becomes a vector | `[0.1, −0.2, 0.3]` |
| **2. Masked self-attention** [10] | Can only see itself, since the mask hides the later words | nothing new learned |
| **3. Cross-attention** [11] | Mixes in the encoder's notes | I 0.33 · like 0.33 · soccer 0.34 |
| **4. Feed forward** [6] | Processes that mix | — |
| **5. Linear + Softmax** [12–13] | Gives a probability for each Spanish word | see the next table |

**The guess at slot 1** (random weights, so it's wrong):

| Spanish word | me | gusta | el | fútbol | `</s>` |
|---|---|---|---|---|---|
| **Probability** | **0.19** ✗ | 0.23 | 0.14 | **0.28** ← guess | 0.17 |

**Then the model learns from the mistake:**

| Step | What happens |
|---|---|
| **6. Loss** | Correct answer is **"me"**, but P("me") = 0.19, so the loss = −log(0.19) ≈ **1.7** (high) |
| **7. Backprop** | Every weight is nudged a little to raise P("me"):<br><br>**Linear** scores "me" higher<br><br>**Cross-attention** looks more at "I" and "like"<br><br>**Encoder** makes those notes more useful |

## The other slots, at the same time

| Slot | Sees (thanks to the mask) | Target | Nudged towards… |
|---|---|---|---|
| 2 | `<s> me` | gusta | "like" note → "gusta" |
| 3 | `<s> me gusta` | el | "soccer" note → "el" |
| 4 | `<s> me gusta el` | fútbol | "soccer" note → "fútbol" |
| 5 | `<s> me gusta el fútbol` | `</s>` | sentence done → stop |

The 5 losses are **averaged**, and **one** backpropagation updates the weights for all of them.

## Repeat

| After… | P("me") at slot 1 |
|---|---|
| 1 step | 0.19 → 0.21 |
| 100 steps | 0.70 |
| many epochs | **0.95** |

> **Two things to remember**
> - With **one** sentence pair, the model would just **memorise** it. The general pattern "I like → me gusta" comes from **thousands** of different pairs, which is why Problem 2 trains on a whole dataset.
> - The model never stores a rule like "I = me". It learns: **"at slot 1, when the notes look like *I like…*, output *me*."**
