# mini-llm-java

A small GPT-style language model, written from scratch in plain Java: no machine-learning libraries, no dependencies. Matrix math, attention, layer norm, backpropagation, the Adam optimizer and multithreaded training are all hand-written in this repo.

It predicts the next **character**, so it learns spelling, words and a little style from whatever text file you give it. It is a learning project, not a chatbot: it **continues** the text you type in the style of its training data and cannot answer questions (see [Limitations](#limitations)).

## Quick start

Needs JDK 11 or newer. All commands run from the `src/` folder.

```
cd src
javac *.java
java GradCheck
java ChatTransformer
```

- `java GradCheck` checks that the math is right (see [How it is tested](#how-it-is-tested)). It should print `(OK)` on every result line.
- `java ChatTransformer` trains for 7 minutes on `shakespeare.txt` (included), prints a progress line every minute, saves the best model, then opens a chat prompt.
- `java ChatTransformer chat` chats with the last saved model without training. Type `exit` to quit.

To train on your own text, put a plain `.txt` file next to the code:

```
java ChatTransformer data=mytext.txt chars=2000000 seconds=900 name=mine
java ChatTransformer chat name=mine
```

The model is saved as `mine-best.bin` and `mine-last.bin`. Model files and `tinystories*.txt` are git-ignored.

### Options

Give them as `key=value` after `ChatTransformer`.

| Option | Default | Meaning |
|---|---|---|
| `data` | `shakespeare.txt` | Text file to learn from |
| `chars` | 200000 | How many characters of it to train on |
| `seconds` | 420 | Training time limit |
| `windows` | off | Train for exactly N windows instead of a time limit (fair comparisons across machines) |
| `dim` | 64 | Model width (24 gives the small 24k-parameter model) |
| `layers` / `heads` | 3 / 4 | Transformer blocks / attention heads (`dim` must divide by `heads`) |
| `context` | 32 | Characters the model sees at once |
| `batch` | 16 | Windows per update (their gradients are averaged) |
| `threads` | all cores | CPU threads to use. On my 10-core Mac, speed stopped improving at 8 |
| `lr` | 0.001 | Adam learning rate at the start; it decays linearly to 10% |
| `valchars` / `valstart` | 5000 / right after the training text | Size and position of the held-out passage used for scoring |
| `name` | none | Prefix for the saved model files |

`dim=24 batch=1 threads=1` gives the original small model.

## How it works

- **Tokenizer:** every distinct character in the data is one token. Characters seen fewer than 3 times are dropped, and curly quotes and dashes are turned into plain ones.
- **Model:** learned character embeddings plus sinusoidal position encodings, then a stack of blocks. Each block has multi-head **causal** self-attention (a position can never see the future) and a ReLU feed-forward layer, each with a residual connection and layer norm. A final linear layer produces a score for every character. The default is 3 blocks, 4 heads, width 64: about 156,000 parameters.
- **Training:** random windows of 33 characters; at **every** position the model predicts the next character, and the loss is the average over positions. Gradients of a batch of windows are computed on several threads at once, averaged, and applied with one Adam step.
- **Sampling:** temperature 0.8.
- **Scoring:** after each minute the model is scored on a held-out passage that it never trains on. The progress line also shows `train-eval` (the same score on a slice of the training text), `real words` (share of generated words that appear in the data file) and `copied` (share of generated 20-character pieces found verbatim in the training text, a memorization check).

### Files in `src/`

| File | Role |
|---|---|
| `Matrix.java`, `Vector.java` | Hand-written tensor math |
| `Adam.java`, `Param.java` | Optimizer, and one block of weights with its gradient |
| `AttentionLayer.java`, `FeedForward.java`, `LayerNorm.java` | The model's building blocks (forward and backward) |
| `MultiLayerTransformer.java` | The full model: training, evaluation, sampling, save and load |
| `ChatTransformer.java` | Command-line program: data cleaning, training loop, reports, chat |
| `GradCheck.java` | Correctness checks |
| `Benchmark.java` | Speed test: `java Benchmark 64 16 8` (dim, batch, threads) |

## How it is tested

`java GradCheck` runs these checks:

1. **Causality:** changing a later character must not change any earlier position.
2. **Gradients:** the backpropagation gradients are compared with numerical derivatives (finite differences), for a single window and for a batch computed on 3 threads.
3. **Threads:** one thread and three threads must give the same weights (they agree to about 1e-15) and the same evaluation result.

## Results

All numbers are single runs on one Mac (7 minutes each), scored on a held-out passage. Lower loss is better; the loss is the average cross-entropy in nats per character, and random guessing would score about 4.2. **Losses are only comparable on the same data and the same held-out passage.**

| Data | Model | Held-out loss | Real words |
|---|---|---|---|
| Shakespeare (200k characters) | 24k parameters | 1.619 | 83% |
| TinyStories (2.5M characters) | 24k parameters | 1.297 | 85% |
| TinyStories (2.5M characters) | 156k parameters, batch 16 | **1.242** | 95% |

Lessons so far:

- More training text helps at equal compute: on a fixed held-out passage, 200k, 500k and 1M characters of Shakespeare scored 2.143, 2.010 and 1.828 (three 10-minute runs of the 24k-parameter model, about 190,000 windows each).
- The 24k-parameter model was capacity-limited: its training and held-out scores were equal and still falling. The 156k-parameter model reached a lower loss from fewer windows.
- A larger batch needs a different learning rate: 0.003 works for the small model but breaks the larger one, so the default stays 0.001.
- The `real words` and `copied` percentages come from one 1,000-character sample, so differences of a few points are noise.

Sample from the 156k-parameter model on TinyStories (temperature 0.8):

> They happy to find the slide." Tom Tom took a peeppy." Lily free! Mommy said, "Let's hand day both was sad and they said, ...

Most words are real and the style is recognizable, but the sentences are not grammatical.

## Limitations

- It continues text; it does not understand questions. Typing "Hi" gives a continuation, not a reply. Conversation would need training on dialogue data, which is not done yet.
- A model this small trained for minutes learns words, not grammar. Fluent English needs far more parameters, data and compute than pure Java on a laptop provides.
- Speed: about 1,100 training windows per second on a 10-core Mac with 8 threads. Each window allocates a lot of short-lived memory, which probably limits scaling beyond that; I have not profiled it.
- With several threads, floating-point addition order varies, so two runs with identical settings differ slightly. Time-limited runs were never exactly repeatable anyway; use `windows=N` and `threads=1` for the most repeatable run.

## Development log

- **Step 20:** fixed the training fundamentals: predict at every position, causal mask, Adam, random windows, learning-rate decay, a held-out score at fixed positions, numerical gradient checks.
- **Step 21:** configurable data pipeline (any text file, cleaning, fixed held-out passage), plus the `real words` and `copied` diagnostics.
- **Step 22:** multithreaded mini-batch training, a bigger default model, and the `windows=` budget.

## Possible next steps

- Train on dialogue-formatted text so the model can answer in a conversation format.
- Reduce memory allocation in the matrix code to scale better across cores.
