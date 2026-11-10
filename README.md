# mini-llm-java

A small GPT-style language model, written from scratch in plain Java: no machine-learning libraries, no dependencies. Matrix math, attention, layer norm, backpropagation, the Adam optimizer, a word-piece (BPE) tokenizer and multithreaded training are all hand-written in this repo.

It predicts the next **token**: one character by default, or a learned word-piece with `vocab=N`. It learns spelling, words and a little style from whatever text file you give it. It is a learning project, not a chatbot. Trained on ordinary text it **continues** what you type in the style of its training data. Trained on chat transcripts (Steps 24 to 27) it writes short replies in a `User:` / `Bot:` format. The replies have the right shape but are often off the point and sometimes garbled (see [Chat results](#chat-results) and [Limitations](#limitations)).

## Quick start

Needs JDK 11 or newer. All commands run from the `src/` folder.

```
cd src
javac *.java
java GradCheck
java TokenizerCheck
java ChatTransformer
```

- `java GradCheck` and `java TokenizerCheck` check that the math and the tokenizer are right (see [How it is tested](#how-it-is-tested)). Every result line should end in `(OK)`.
- `java ChatTransformer` trains for 7 minutes on `shakespeare.txt` (included), prints a progress line every minute, saves the best model, then opens a chat prompt.
- `java ChatTransformer chat` chats with the last saved model without training. Type `exit` to quit.

To train on your own text, put a plain `.txt` file next to the code:

```
java ChatTransformer data=mytext.txt chars=2000000 seconds=900 name=mine
java ChatTransformer chat name=mine
```

To use word-piece tokens instead of single characters, add `vocab=512`:

```
java ChatTransformer data=mytext.txt chars=2000000 vocab=512 name=minebpe
```

The model is saved as `mine-best.bin` and `mine-last.bin`. Model files, `tinystories*.txt`, `dialogue*.txt` and `chitchat*.txt` are git-ignored.

### Chat format (Steps 24 to 27)

The chat data is not in the repo. Two small converters download it from Hugging Face (they need an internet connection) and write `User:` / `Bot:` transcripts. Each prints the exact training command, including the right `chars=` and `valstart=` for your download (the numbers below are from one run; use the ones your converter prints):

```
java ChitChatData
java ChatTransformer data=chitchat.txt chars=5573713 valstart=5573720 valchars=20000 vocab=512 context=96 seconds=900 threads=8 name=chatbot loss=bot
java DialogueChat name=chatbot data=chitchat.txt chars=5573713
java ReplyCheck name=chatbot data=chitchat.txt chars=5573713
```

- `java ChitChatData` writes `chitchat.txt` from small-talk chats. `limit=` and `testlimit=` set how many training and held-out conversations to read (default 4000 and 150).
- `java DialogueData` writes `dialogue.txt` from a different set of short assistant-style chats (the Step 24 data).
- `java DialogueChat` asks a trained model 8 fixed questions at two temperatures, then lets you chat (`reset` starts a new conversation, `exit` quits). `temp=0.3` gives more careful replies; `probes=0` skips the fixed questions.
- `java ReplyCheck` measures whether the model's replies depend on what the user said (see [How it is tested](#how-it-is-tested)).

Data sources: [HuggingFaceTB/everyday-conversations-llama3.1-2k](https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k) (Apache-2.0) and Google's [Synthetic-Persona-Chat](https://huggingface.co/datasets/google/Synthetic-Persona-Chat) (CC BY 4.0).

### Options

Give them as `key=value` after `ChatTransformer`.

| Option | Default | Meaning |
|---|---|---|
| `data` | `shakespeare.txt` | Text file to learn from |
| `chars` | 200000 | How many characters of it to train on |
| `seconds` | 420 | Training time limit |
| `windows` | off | Train for exactly N windows instead of a time limit (fair comparisons across machines) |
| `vocab` | 0 | Tokenizer size. 0 = one token per character. `vocab=512` = the characters plus about 440 learned word-pieces |
| `dim` | 64 | Model width (24 gives the small 24k-parameter model) |
| `layers` / `heads` | 3 / 4 | Transformer blocks / attention heads (`dim` must divide by `heads`) |
| `context` | 32 | Tokens the model sees at once (about 32 characters with `vocab=0`, about 90 with `vocab=512`) |
| `batch` | 16 | Windows per update (their gradients are averaged) |
| `threads` | all cores | CPU threads to use. On my 10-core Mac, speed stopped improving at 8 |
| `lr` | 0.001 | Adam learning rate at the start; it decays linearly to 10% |
| `valchars` / `valstart` | 5000 / right after the training text | Size and position of the held-out passage used for scoring |
| `loss` | `all` | `all` = every character is scored. `bot` = for `User:` / `Bot:` data, only the bot's reply (the text after `Bot:` up to the end of its line) is trained on and scored; the model still reads the user's lines |
| `name` | none | Prefix for the saved model files |

`dim=24 batch=1 threads=1` gives the original small model.

## How it works

- **Cleaning:** curly quotes and dashes are turned into plain ones, `<|endoftext|>` markers become newlines, and characters seen fewer than 3 times are dropped.
- **Tokenizer:** with `vocab=0` every distinct character is one token. With `vocab=N` it starts from the characters and learns merges (byte-pair encoding) from the training text until it has N tokens. Merges never cross a word, a number, a punctuation group or a whitespace run, learning is deterministic, and encoding then decoding gives back exactly the same text. The tokenizer is saved inside the model file.
- **Model:** learned token embeddings plus sinusoidal position encodings, then a stack of blocks. Each block has multi-head **causal** self-attention (a position can never see the future) and a ReLU feed-forward layer, each with a residual connection and layer norm. A final linear layer produces a score for every token. The default is 3 blocks, 4 heads, width 64: about 156,000 parameters with characters, 214,000 with `vocab=512` (the extra ones are almost all in the embedding and output layers).
- **Training:** random windows of 33 tokens; at **every** position the model predicts the next token, and the loss is the average over positions. Gradients of a batch of windows are computed on several threads at once, averaged, and applied with one Adam step.
- **Reply-only training (`loss=bot`):** the model still reads every token, but only the predictions inside a bot reply count towards the loss and the gradient (averaged over those positions). A reply runs from just after `Bot:` to the end of its line, so the model also learns where a reply ends. The loss columns of the progress line then measure reply tokens only, which is why they cannot be compared with a `loss=all` run; use `ReplyCheck` for that.
- **Sampling:** temperature 0.8.
- **Scoring:** after each minute the model is scored on a held-out passage that it never trains on. Losses are always reported **per character** (the total loss divided by the number of characters the scored tokens stand for), so runs with different `vocab` settings can be compared on the same passage. The progress line also shows `train-eval` (the same score on a slice of the training text), `real words` (share of generated words that appear in the data file) and `copied` (share of generated 20-character pieces found verbatim in the training text, a memorization check).

### Files in `src/`

| File | Role |
|---|---|
| `Matrix.java`, `Vector.java` | Hand-written tensor math |
| `Adam.java`, `Param.java` | Optimizer, and one block of weights with its gradient |
| `AttentionLayer.java`, `FeedForward.java`, `LayerNorm.java` | The model's building blocks (forward and backward) |
| `Tokenizer.java` | Character or byte-pair tokenizer: train, encode, decode, save, load |
| `MultiLayerTransformer.java` | The full model: training, evaluation, sampling, save and load |
| `ChatTransformer.java` | Command-line program: data cleaning, training loop, reports, chat |
| `GradCheck.java`, `TokenizerCheck.java` | Correctness checks |
| `DialogueData.java`, `ChitChatData.java` | Download chat datasets and write `User:` / `Bot:` training files |
| `DialogueChat.java` | Test questions and chat for a model trained on those files |
| `ReplyCheck.java` | Measures whether replies depend on the user's message |
| `OverlapCheck.java` | Baseline for `copied`: `java OverlapCheck tinystories.txt 2500000` (data file, training characters; use the same as the training run) |
| `Benchmark.java` | Speed test: `java Benchmark 64 16 8 15 512` (dim, batch, threads, seconds, vocab; all optional) |

## How it is tested

`java GradCheck` runs these checks, once on a character model and once on a small word-piece model:

1. **Causality:** changing a later token must not change any earlier position.
2. **Gradients:** the backpropagation gradients are compared with numerical derivatives (finite differences), for a single window and for a batch computed on 3 threads.
3. **Threads:** one thread and three threads must give the same weights (they agree to about 1e-15) and the same evaluation result.
4. **Checkpoints:** saving a model and loading it back must give the same outputs.
5. **Reply masks (`loss=bot`):** a mask that marks every position gives exactly the unmasked result; a mask that marks one position scores exactly that position; the masked gradients match numerical derivatives on 3 threads (a window with nothing marked adds nothing); `evaluateTokens` scores only the marked tokens; and in a transcript exactly the reply characters are marked, with no token straddling a reply boundary, for both tokenizers.

A complete run prints 26 lines ending in `(OK)` and no `FAIL`.

`java TokenizerCheck` checks that decoding gives back exactly what was encoded (on the training text and on unseen text), that the token count matches what training counted, that unknown characters are skipped, that training twice gives identical merges, that save and load change nothing, and that without merges every character is its own token.

I also broke the code on purpose (a wrong mask, a wrong gradient) to confirm that the checks fail when they should.

`java ReplyCheck name=... data=... chars=...` takes a trained model and the held-out part of a chat file (everything after the `chars` training characters). For each held-out user line and bot reply it scores the real reply twice: after the real user line, and after a different, randomly chosen user line with the same earlier lines. Both scores are nats per character of the reply; the model is not trained or changed. A model that uses the question scores the real reply better than the random one, and the difference is reported with its standard error. It was checked on a toy chat (questions about animal sounds and colours, each with one fixed answer; a 27k-parameter model): after a single training step the gain is 0.002 +/- 0.001 and the real question wins in 41% of pairs, and after 6,000 training windows it is 0.343 +/- 0.016 and wins in 100% of pairs.

## Results

All numbers are single runs on one Mac (7 minutes each, except the last row, which ran for 20), scored on a held-out passage. Lower loss is better; the loss is the average cross-entropy in nats per character, and random guessing would score about 4.2 with single characters and about 2.3 with `vocab=512` (guessing among 512 tokens that stand for about 2.7 characters each). **Losses are only comparable on the same data and the same held-out passage.**

| Data | Model | Held-out loss | Real words |
|---|---|---|---|
| Shakespeare (200k characters) | 24k parameters | 1.619 | 83% |
| TinyStories (2.5M characters) | 24k parameters | 1.297 | 85% |
| TinyStories (2.5M characters) | 156k parameters, batch 16 | 1.242 | 95% |
| TinyStories (2.5M characters) | 214k parameters, `vocab=512`, batch 16 | 1.009 | 93% |
| TinyStories (2.5M characters) | same, trained for 20 minutes | **0.923** | 98% |

Lessons so far:

- More training text helps at equal compute: on a fixed held-out passage, 200k, 500k and 1M characters of Shakespeare scored 2.143, 2.010 and 1.828 (three 10-minute runs of the 24k-parameter model, about 190,000 windows each).
- The 24k-parameter model was capacity-limited: its training and held-out scores were equal and still falling. The 156k-parameter model reached a lower loss from fewer windows.
- A larger batch needs a different learning rate: 0.003 works for the small model but breaks the larger one, so the default stays 0.001.
- Word-piece tokens (`vocab=512`) lowered the held-out loss from 1.242 to 1.009 in the same 7 minutes. Two things changed at once, so this run cannot say which one mattered: a window now covers about 90 characters instead of 32 (the model saw about 2.5 times more text in the same time), and the model has about 57,000 more parameters. The two tokenizer modes also score slightly different character positions; I did not measure how much that matters.
- Training the `vocab=512` model for 20 minutes instead of 7 (about 26 passes over the 2.5M characters instead of 11) lowered the held-out loss from 1.009 to 0.923. The training loss ended at 0.907, close to the held-out 0.923 (the two are scored slightly differently), so the model was not memorizing the training text; it was still improving, slowly (about 0.01 in the last four minutes, partly because the learning rate decays toward the end of the time limit). The two runs also differ in how long that decay lasts.
- The `real words` and `copied` percentages come from one 1,000-character sample per report, so differences of a few points are noise. In the 20-minute run `copied` ranged from 3% to 24% and was higher in the last reports (24%, 15%, 17%) than in the first (3% to 6%). To judge that, `java OverlapCheck` measures how often real text the model never saw repeats 20-character pieces of the training text: 14.4% over a 50,000-character passage after the training text, and 3% to 23% across its 1,000-character samples (median 14%). The model's `copied` values are in that range (one report, 24%, is slightly above the highest of the 50 real samples, 22.8%), so there is no clear sign of memorization, only the normal reuse of stock phrases; one 1,000-character sample per report cannot settle it.
- In the `vocab=512` runs the `train-eval` score is higher than the held-out score only because they are different passages; it does not show whether the model is memorizing.

Sample from the 20-minute `vocab=512` model on TinyStories (temperature 0.8, excerpt):

> They make friends inside to play with their toys at it and came pick. They would play on the swings - the shorpt in the grass. They put the cat on the floor sing and started to change. ...

Most words are real and the sentences have the right shape, but the grammar and meaning break within a sentence, and invented words still appear ("shorpt", and "cherne" and "creat" elsewhere in the same sample).

## Chat results

Single runs of the 214k-parameter model (`dim=64 layers=3 heads=4 vocab=512 context=96`) on chat data. Losses are per character and only comparable on the same data.

| Data | Training | Held-out loss | Real words | `copied` (baseline) |
|---|---|---|---|---|
| everyday-conversations (2,260 training chats) | 10 min | 1.343 | 71% | 14% (37.2%) |
| Synthetic-Persona-Chat (4,000 chats, 5,573,713 training characters) | 15 min, `loss=all` | 0.765 | 89% | 41% (65.4%) |

In the table, `copied` is the share of generated 20-character pieces found verbatim in the training text; the figure in brackets is the same measurement on real held-out dialogue, printed by the converter. Chat data repeats stock phrases a lot, so that baseline is much higher than for stories.

**Step 24 (assistant-style chats): format learned, conversation not.** The bot's first reply is the same greeting in 2,248 of the 2,260 training chats, and the data contains no "how are you" at all. In the 8 test questions at two temperatures (16 replies), 13 replies were that same greeting, whatever was asked. The model had learned the format and a fixed greeting, not conversation.

**Step 25 (small talk): same model size, better data.** The 4,000 small-talk chats open in many different ways and do contain "how are you" exchanges. After 15 minutes the held-out loss was 0.765 (train-eval 0.814, so held-out text was no harder than training text), about 89% of generated words were real words, and `copied` was 41% against the 65.4% baseline.

**Step 26: do the replies depend on the question?** `ReplyCheck` on up to 400 held-out pairs:

| Model | Reply loss, real question | Reply loss, random question | Gain |
|---|---|---|---|
| 15 min, `loss=all` | 0.862 | 0.982 | 0.119 +/- 0.007 (12% lower; real question wins in 91% of pairs) |
| 15 min, `loss=bot` | 0.850 | 0.969 | 0.118 +/- 0.009 (12% lower; 91%) |

The model does use the question, but weakly. The gain is strongest on the opening exchange (0.339 +/- 0.100 for the `loss=bot` model) and smaller elsewhere.

**Step 27: training on the replies only.** `loss=bot` changed nothing measurable: the reply loss moved from 0.862 to 0.850 (about 1%) and the gain stayed at 0.118. That run also got through about 20% more windows in the same 15 minutes (204,384 against 170,832); the code skips the softmax at positions that are not scored. On the toy chat it helped a lot: after the same 2,000 windows the reply loss was 1.361 against 2.283 and the gain 0.102 +/- 0.010 against 0.038 +/- 0.010, so the mechanism works, but that did not carry over to the real data.

Some of the test questions at temperature 0.3, one sample each (an impression, not a measurement; `ReplyCheck` says the two models are about equal):

| You | `loss=all` | `loss=bot` |
|---|---|---|
| How are you? | I'm a teacher. | I'm doing well, thanks! |
| What is your name? | I love to read, go to the movies, and spend time with my friends. | Hi, I'm Sam. |
| Thank you! | You will! It's a great way to relax. | Hello! |
| Goodbye! | So, what are you doing? | Hi! |

The `loss=bot` model answered both "Can you help me?" and "What is the capital of France?" with `I'm not sure yet.`

## Limitations

- It does not understand questions. Trained on stories, typing "Hi" gives a continuation, not a reply. Trained on chat data it answers greetings (and sometimes "How are you?") sensibly, and otherwise its replies often ignore what you said, repeat a stock phrase (`I'm not sure yet.`) or are garbled. It has no facts: asked for the capital of France it answers `I'm not sure yet.` or nonsense. The chat data comes from many different invented personas, so it has no consistent personality.
- A model this small trained for minutes learns words, not grammar. Fluent English needs far more parameters, data and compute than pure Java on a laptop provides.
- Word-pieces are not words: the model can still put pieces together into things that are not words.
- The tokenizer only knows the characters and merges seen in its training text; any other character is skipped.
- Speed: about 1,100 training windows per second with characters and about 630 to 740 with `vocab=512` (but each of those windows covers about 2.7 times more text), on a 10-core Mac with 8 threads. The chat runs (`context=96`, `vocab=512`) ran at 188 windows per second, or 225 with `loss=bot`. Each window allocates a lot of short-lived memory, which probably limits scaling beyond that; I have not profiled it.
- With several threads, floating-point addition order varies, so two runs with identical settings differ slightly. Time-limited runs were never exactly repeatable anyway; use `windows=N` and `threads=1` for the most repeatable run.

## Development log

- **Step 20:** fixed the training fundamentals: predict at every position, causal mask, Adam, random windows, learning-rate decay, a held-out score at fixed positions, numerical gradient checks.
- **Step 21:** configurable data pipeline (any text file, cleaning, fixed held-out passage), plus the `real words` and `copied` diagnostics.
- **Step 22:** multithreaded mini-batch training, a bigger default model, and the `windows=` budget.
- **Step 23:** byte-pair word-piece tokenizer (`vocab=N`), losses reported per character so the two modes can be compared, a new model-file format (older character-only files still load), and tokenizer and checkpoint tests. First result: held-out loss 1.242 to 1.009 on TinyStories in 7 minutes, and 0.923 in 20, with the caveats above. `OverlapCheck.java` measures the baseline for the `copied` number: real unseen text scored 14.4%, the same range as the model's.
- **Step 24:** chat-format data. `DialogueData` turns a small set of assistant-style chats into `User:` / `Bot:` transcripts and `DialogueChat` tests a trained model with fixed questions. Result: the format and a fixed greeting were learned, nothing more.
- **Step 25:** `ChitChatData` builds the same format from Synthetic-Persona-Chat small talk, because the Step 24 data has one fixed greeting and no "how are you". Held-out loss 0.765 per character after 15 minutes.
- **Step 26:** `ReplyCheck` measures whether replies depend on the question (reply loss after the real user line against after a random one). Gain 0.119 +/- 0.007: weak but real.
- **Step 27:** `loss=bot` trains and scores only the bot's replies, with the mask tests added to `GradCheck`. No measurable change in `ReplyCheck` on the real data.

## Possible next steps

- Test whether scale raises the `ReplyCheck` gain: a larger model (`dim=96 layers=4 heads=4`, about 540k parameters) trained for 40 minutes against the 15-minute baseline above.
- Check whether the chat data contains goodbyes and thanks; the model answers those with greetings.
- Reduce memory allocation in the matrix code to scale better across cores.
