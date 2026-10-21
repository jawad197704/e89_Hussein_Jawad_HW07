# How "I like soccer" becomes "me gusta el fútbol" — the whole story in 4 steps

## Step 1 — Encoder: understand the English (runs **once**)

"I like soccer" → token IDs `[4, 5, 6]` → vectors → attention + feed-forward (× N)

**Result:** 3 vectors, one per English word, each now carrying context. These are the encoder's **notes** (`memory` in PyTorch):

| Word | What its note now says |
|---|---|
| I | the subject |
| like | verb: *"enjoy"* (not *"similar to"*) |
| soccer | a sport, the thing being liked |

## Step 2 — Decoder: write the Spanish, **one word per run**

On every run the decoder:
1. looks at **what it has written so far** (masked self-attention [10]),
2. checks the encoder's **notes** for what matters now (cross-attention [11]),
3. scores every Spanish word and **picks the most likely**,
4. **adds** that word, then runs again, until it outputs `</s>` (end).

| Run | Written so far | Mostly checks the note for… | Picks |
|---|---|---|---|
| 1 | `<s>` | "I", "like" | **me** |
| 2 | `<s> me` | "like" | **gusta** |
| 3 | `<s> me gusta` | "soccer" | **el** |
| 4 | `<s> me gusta el` | "soccer" | **fútbol** |
| 5 | `<s> me gusta el fútbol` | (nothing left) | **`</s>`** → stop |

*Note:* the decoder doesn't swap words one for one. Spanish says "I like" as **"me gusta"** ("it pleases me"), and the decoder learns that pattern.

## Step 3 — "Outputs (shifted right)": the decoder's input

The decoder's job at every slot is: **"given the words so far, guess the next word."** To set this up, the Spanish sentence is moved **one slot to the right**, and `<s>` fills the first slot:

| Slot | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| **Decoder input** (shifted right) | `<s>` | me | gusta | el | fútbol |
| **Must predict** | me | gusta | el | fútbol | `</s>` |

**Why shift?** Without it, slot 1 would show "me" *and* ask for "me", so the model would learn to **copy** instead of **predict**.

## Step 4 — How it learns this (training)

At first the weights are **random**, so every Spanish word gets about the same tiny probability. Training repeats a simple loop over thousands of sentence pairs:

| | What happens | Example (slot 1) |
|---|---|---|
| **Guess** | Decoder scores every Spanish word | P("me") = 0.0001 |
| **Check** | Loss compares the guess with the correct word | correct = "me" → big loss |
| **Adjust** | Backpropagation nudges **all** weights a little | P("me") goes up next time |

All 5 slots are trained **at once**: the whole shifted sentence goes in together, and the mask [10] stops each slot from peeking ahead. After many epochs, P("me") ≈ **0.95** whenever the notes say "I like…".

> **In one line:** the encoder turns the English into context-rich notes once; the decoder then writes Spanish one word at a time, each time asking "given what I've written and the notes, what comes next?", a skill learned by guessing, checking and adjusting over thousands of examples.
