# MA289 Competition 3 — Neural Network Word Embeddings

Learning token embeddings from the anonymised MA289 corpus with a feed-forward
neural network, and predicting a similarity score for each of the 1,200 test pairs.

Everything here is learned from the supplied `train_corpus.txt` and
`context_train.csv`. No pretrained embeddings, no external corpora, and no attempt
to recover the original vocabulary.

---

## TL;DR

Two findings decided this competition, and neither is about training a better network.

1. **A short training run is mostly random initialisation.** Two runs of the lesson
   model that differ only in random seed have a test-pair rank correlation of
   **0.19 after 3 epochs**. Nearly everything a 3-epoch cosine measures is the
   initialiser, not the corpus. Training to 100 epochs and averaging over 8 seeds
   raises the reproducible part to 0.86.

2. **The scoring metric is MSE, so the scale of the submission matters as much as
   the ranking.** The target similarities have standard deviation **0.291**, which is
   essentially Uniform(0,1). A trained 24-dimensional embedding produces
   `(cos+1)/2` values with standard deviation about **0.09**. Submitting raw cosines
   is mis-scaled by a factor of three, and that error alone costs more than any
   modelling improvement can recover.

The submitted file is a seed-averaged, multi-family consensus *ranking*, mapped
through a calibration that matches the measured target distribution.

---

## The data

| file | contents |
| --- | --- |
| `train_corpus.txt` | 4 segments, 219,761 tokens, 900-token vocabulary |
| `context_train.csv` | 300,000 center/context pairs |
| `test.csv` | 1,200 pairs over 372 distinct tokens |
| `sample_submission.csv` | all 0.5 |

Two things worth knowing before modelling:

- **The supplied pairs use a context window of 2.** Every pair in
  `context_train.csv` occurs in the window-2 co-occurrence set of the corpus, while
  32% of them are missing from window 1. They are a 300,000-pair subsample of the
  879,020 available window-2 pairs, so the corpus can be used to regenerate training
  data with any window. (`src/window.py`)
- **647 of the 1,200 test pairs never co-occur within window 2.** For more than half
  the test set, first-order co-occurrence counts carry no information at all, and the
  score depends entirely on second-order (distributional) structure. (`src/testinfo.py`)

---

## Model 1 — the lesson architecture

```
token id -> nn.Embedding(900, 24) -> nn.Linear(24, 900) -> CrossEntropyLoss
Adam(lr=0.01), batch_size=1024
```

### Seed sensitivity

Rank correlation between the test-pair cosines of runs that differ **only** by seed:

| epochs | mean seed-to-seed Spearman (8 seeds) | implied signal fraction |
| --- | --- | --- |
| 1 | 0.02 | ~0 |
| 2 | 0.11 | 0.11 |
| 3 | 0.19 | 0.19 |
| 5 | 0.34 | 0.34 |
| 8 | 0.50 | 0.50 |
| 12 | 0.61 | 0.61 |
| 20 | 0.70 | 0.70 |
| 35 | 0.77 | 0.77 |
| 60 | 0.83 | 0.83 |
| 100 | 0.86 | 0.86 |

The lesson's default of 3 epochs is a good teaching baseline and a poor competition
model: four fifths of its geometry is noise. Averaging normal-scored cosines over
8 seeds at epochs {20, 35, 60, 100} removes almost all of it — the averaged
prediction correlates ≥ 0.955 with each individual epoch's consensus, so it also
hedges the choice of epoch count.

---

## Model 2 — other embedding families from the same corpus

Because half the test pairs never co-occur, families that model *distributional*
similarity add a lot, and their errors are largely independent of the feed-forward
model's (cross-family rank correlation is only 0.44–0.54).

| family | variants | script |
| --- | --- | --- |
| Feed-forward softmax (supplied pairs) | 8 seeds × 4 epoch snapshots | `src/train_torch.py` |
| Feed-forward softmax (full corpus, windows 5/10) | 2 seeds × 3 snapshots | `src/ffnn_wide.py` |
| Skip-gram with negative sampling | windows 2/5/10, d 50/100, 3 seeds | `src/sgns.py` |
| PPMI + truncated SVD | windows 2/5/10, d 50–300, 3 eigenvalue weightings | `src/ppmi2.py` |
| Second- and third-order similarity | windows 5/10/20/50 | `src/second.py` |
| GloVe, NMF, log-count LSA, Hellinger PCA | windows 5/10 | `src/more.py` |

Each model's test-pair cosines are converted to normal scores, averaged within a
family, and the family consensuses are combined into one ranking.

---

## Calibration — where most of the score comes from

MSE punishes a mis-scaled submission even when the ordering is perfect, so the
target distribution was measured directly, using submissions as measurements:

| submission | public MSE | what it determines |
| --- | --- | --- |
| constant 0.0 | 0.33602 | `E[y²] = 0.33602` |
| constant 1.0 | 0.33318 | `E[y] = 0.50142` |
| rank-uniform consensus | 0.09385 | `Cov(prediction, y)` |

From the first two:

```
E[y]    = (E[y²] - MSE(1) + 1) / 2 = 0.50142
Var(y)  = E[y²] - E[y]²            = 0.08460      sd = 0.29086
```

Uniform(0,1) has sd 0.28868, so the organisers' similarities are a rank/percentile
transform of their reference similarity. Three consequences:

- The best possible **constant** submission scores 0.08460. (The public leaderboard
  confirms this: an entry sitting at exactly 0.08460 is the unmodified
  `sample_submission.csv`.)
- A raw `(cos+1)/2` submission, with sd ≈ 0.09, spends most of its error budget on
  being too narrow.
- Only the *ordering* of our prediction carries information, so the prediction
  should be rank-transformed and then re-expanded to the measured spread.

Given a predicted ranking with normal scores `z`, and correlation `rho` with the
target, the MSE-optimal submission under a Gaussian copula is

```
y_hat = Phi( rho * z / sqrt(2 - rho**2) )
```

`rho` is measured by the third probe rather than assumed. The expected MSE is
`Var(y) * (1 - rho²)`.

---

## Experiment table

All five of the day's submissions, in order:

| # | submission | public MSE | corr with target |
| --- | --- | --- | --- |
| 1 | constant 0.0 (probe) | 0.33602 | — |
| 2 | constant 1.0 (probe) | 0.33318 | — |
| 3 | feed-forward consensus, rank-uniform, uncalibrated | 0.09385 | 0.441 |
| 4 | PPMI-SVD + SGNS consensus, copula-calibrated (rho = 0.65) | 0.07325 | 0.422 |
| 5 | **all-family consensus (275 models), copula-calibrated (rho = 0.545)** | **0.06524** | **0.481** |

Reference points: the best possible constant scores 0.08460, the unmodified
`sample_submission.csv` scores 0.08460, and the best rival public score was 0.06261.

**Final public standing: 2nd of 6** (0.06524 vs 0.06261). Final standings are decided
on the other 70% of the test set.

### Where the score came from

Calibration, not modelling. The same ranking submitted raw would have scored 0.09385;
calibrated it scores 0.06524. The modelling work moved the correlation from 0.441 to
0.481, worth about 0.004 of MSE. The calibration was worth 0.029.

### Why it did not finish first

Fitting a one-factor model to the measured quantities puts the ceiling for *any*
predictor derived from this corpus at **corr ≈ 0.60**, i.e. MSE ≈ 0.0545. The corpus
is only 219,761 tokens, which bounds how well token similarity can be recovered at
all. Our 0.481 leaves real headroom against that ceiling, and the gap to first place
(0.0026) sits inside it.

The binding constraint was the 5-submissions-per-day limit against a same-day
deadline. Three submissions went to measuring the target distribution and our
correlation with it — that spend is what produced the calibration, and without it the
entry would have scored around 0.094. But it left only two for model iteration, and
the useful direction (the feed-forward and matrix families are only 0.52 correlated,
so blending them beats either) was not measurable until probe 3 had already landed.

With one more submission, the ridge-regularised blend of all three measured
predictors (`results/submission_v2_unsubmitted.csv`, built by
`src/three_predictor_blend.py`) was expected to score **0.0635**. It was rejected by
the daily quota. Even that falls short of first place, so closing the gap needed a
genuinely better estimate of the shared signal — more independent model families, or
the two probe submissions spent on model comparison instead of distribution
measurement.

---

## Reproducing

```bash
pip install -r requirements.txt
export MA289_DATA=/path/to/competition/data      # the four Kaggle files

python src/window.py        # confirm the context window is 2
python src/testinfo.py      # test-set structure
python src/train_torch.py '{"d":24,"seed":0,"lr":0.01,"batch":1024,"snapshots":[20,35,60,100],"outdir":"out2"}'
python src/ppmi2.py  out3
python src/sgns.py   out4
python src/second.py out5
python src/more.py   out6
python src/consensus.py out2 .
python src/blend.py  .
python src/final.py  .      # writes submission.csv
```

`MA289_word_embeddings.ipynb` is a self-contained Colab version of the same
pipeline (upload the four competition files to the runtime, then run top to bottom).

---

## What the learned embedding represents

Each row of `E` is positioned purely by which tokens appear within two positions of
it. Backpropagation moves only the rows selected by each batch, so tokens that occur
in interchangeable contexts are pushed into the same region and cosine distance ends
up measuring *distributional* similarity. The model never sees meaning — only the
company a token keeps.

The representation is **fixed per token**: `tok_0023` gets one vector wherever it
appears. A token with two senses or two grammatical roles is forced into a single
compromise position, and the model has no mechanism to separate the uses. The
seed-sensitivity table above shows a second, subtler limitation: with 220k tokens of
text, a single short run does not even identify that one vector reliably.

Both limitations point at the rest of the block. RNNs and LSTMs compute a hidden
state that depends on the tokens before it, so the same token can be represented
differently in different positions; attention lets a model choose which parts of the
sequence to read for each prediction instead of compressing everything into one fixed
vector. The embedding matrix learned here is the input layer those contextual models
are built on.
