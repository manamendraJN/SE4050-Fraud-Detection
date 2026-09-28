# MLP Model — Working Notes

Running log of decisions, findings, and issues for the MLP component of the
SE4050 fraud detection project. Intended as raw material for the report
(especially **Experimental Design**, **Model Architectures**, and
**Critical Analysis & Discussion**) — add to this after each session rather
than trying to reconstruct it later.

---

## 1. Preprocessing (shared pipeline, not MLP-specific)

- Script: `01_preprocessing.py`, run once, seed = 42, outputs shared via
  Google Drive so all 4 teammates train/evaluate on identical splits.
- Removed 1,081 duplicate rows before splitting.
- Log-transformed `Amount` → `Amount_log` (heavy right skew).
- Stratified 70/15/15 train/val/test split — fraud rate held at ~0.167%
  across all three splits.
- Scaled `Time` and `Amount_log` with `StandardScaler` fit **only on
  training data** (no leakage into val/test).
- Split sizes: Train (198,608, 30), Val (42,559, 30), Test (42,559, 30) — 30 input features (28 PCA components + scaled Time + log-transformed Amount; raw Amount is dropped). Confirmed from the model's first Dense layer (1,984 params = 30 x 64 + 64) and the notebook's printed shapes; an earlier "32" here was likely printed before the raw Amount column was dropped.

**Why it matters for the report:** fair cross-model comparison depends on
every teammate using these exact splits — worth stating explicitly in
Experimental Design as a controlled variable.

---

## 2. MLP Architecture — decisions and rationale

    Dense(64, ReLU) → BatchNorm → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)
    Optimizer: Adam, lr=0.001
    Loss: binary cross-entropy
    Class weights: sklearn compute_class_weight("balanced") — handles ~0.17% fraud imbalance
    Early stopping: monitor=val_auc, mode=max, patience=8, restore_best_weights=True

Things to justify specifically in the report rather than leaving generic:
- Why 64→32 (capacity vs. overfitting risk on a fairly small effective
  positive-class sample)
- Why dropout 0.3 at both layers
- Why early stopping on **val_auc** specifically rather than val_loss —
  AUC is threshold-independent and more meaningful under heavy imbalance
- Why class weighting was chosen over resampling (e.g. SMOTE) — didn't
  test SMOTE, worth naming as a limitation/future work item

---

## 3. Initial Training Results (exploratory run — superseded by the frozen final run in section 4)

From first full run (`02_mlp_model_colab.py`, standalone `!python` execution):

- Training converged well before 50 epochs (early stopping triggered)
- **ROC-AUC: 0.9685**
- **PR-AUC: 0.7279**
- At threshold = 0.5: precision for fraud class = **0.04**, recall = 0.89
  — expected side-effect of class weighting pushing the decision boundary
  down; **not a bug**, but must be explained in the report as the reason
  threshold=0.5 is the wrong operating point for this problem.
- Confusion matrix at threshold=0.5: `[[41050, 1438], [8, 63]]`

**Key point for report:** ROC-AUC and PR-AUC are stable, threshold-based
metrics (precision/recall/F1 at a fixed cutoff) are not, and need a
deliberately chosen operating threshold rather than the default 0.5.

---

## 4. The "threshold = 1.0000" Investigation (important finding)

**What happened:** the original run's threshold analysis (via
`precision_recall_curve` + F1 argmax) reported best threshold = **1.0000**,
F1 = 0.7883, precision = 0.8182, recall = 0.7606. This looked suspicious —
a threshold of exactly 1.0 is not a usable real-world decision boundary.

**Investigation steps:**
1. Reloaded the saved model (`mlp_fraud_model.keras`) and recomputed
   predictions on the test set in a fresh session.
2. Checked the top of the predicted-probability distribution — found the
   model does predict a small cluster of samples at exactly `1.0`
   (34 samples out of 42,559 test rows) due to sigmoid saturation in
   float32.
3. Re-ran `precision_recall_curve` + F1 argmax on the reloaded model: this
   time the best index came out at **threshold ≈ 0.99982** (not exactly
   1.0), F1 = 0.8116, precision = 0.8358, recall = 0.7887 — close to, but
   not identical to, the original run's numbers.

**Conclusion:** the exact "best" threshold is **not stable across
different training runs**. Part of this is a genuine **sample-size
effect**: the test set only contains **71 fraud cases**, so the
F1-vs-threshold curve is being optimized over a very small, noisy set of
positives, and moving even one prediction across the boundary visibly
shifts the "best" threshold.

**Correction:** an earlier version of these notes attributed the
run-to-run drift to floating-point/GPU non-determinism in the *same*
saved model. That was never actually verified. A more likely explanation,
consistent with file timestamps, is that a "Run all" in Colab silently
re-triggered the training cell and overwrote the saved model on Drive —
so successive "reloads" were often not the same model at all, just
successive retrains sharing the same architecture and seed. This doesn't
change the core finding (threshold instability under a tiny positive
class, and the risk of tuning that threshold on the test set), but it
does mean the different threshold/ROC-AUC/PR-AUC values recorded across
sessions reflect **different trained models**, not numerical noise in one
model.

**Methodological fix adopted:** stop selecting the operating threshold
from the **test set** (this is a mild form of test-set leakage — tuning a
decision using the same data used to report final performance). Instead:
select the threshold using the **validation set**, then apply that fixed
threshold once to the test set and report those numbers as final.

    val_proba = model.predict(X_val).ravel()
    val_precisions, val_recalls, val_thresholds = precision_recall_curve(y_val, val_proba)
    val_f1 = 2 * (val_precisions * val_recalls) / (val_precisions + val_recalls + 1e-9)
    best_val_idx = np.argmax(val_f1)
    chosen_threshold = val_thresholds[best_val_idx]

    y_pred_final = (y_proba >= chosen_threshold).astype(int)
    # report classification_report(y_test, y_pred_final, ...) using this fixed threshold

**Why this is good report material:** it's a specific, defensible insight
that most groups working on this exact dataset likely won't catch or
discuss (per the differentiation goal vs. the earlier report reviewed).
Write it up as: what we observed → why it happens → what we changed → why
that's more methodologically sound. This is exactly the kind of reasoning
the Critical Analysis section (30% of grade) rewards.

**Sept 15 exploratory run (superseded — kept here for the record):**

- Threshold selected from validation set: 0.9984
- Validation performance at this threshold: precision=0.8889, recall=0.7887, f1=0.8358
- Test performance at this threshold: precision=0.81, recall=0.79, F1=0.80 (support=71)
- Confusion matrix: `[[42475, 13], [15, 56]]` — 56/71 frauds caught, 13 false positives
- ROC-AUC: 0.9649

This run, the original run (0.9685 ROC-AUC / 0.7279 PR-AUC), and the
from-scratch retrain that was briefly committed on `main` (0.9979
threshold, 0.9595 ROC-AUC, 0.7488 PR-AUC) are three **different trained
models**, not the same model measured three times. Across them, ROC-AUC
has ranged 0.9563–0.9685 (about a 1-point spread) and PR-AUC
0.7225–0.7488 (about a 3-point spread) — contrary to an earlier version
of these notes, this is **not stable across reruns**. That spread is
expected given how few fraud cases the test set has (71) and is itself
worth a line in Critical Analysis: differences under about 5–10 points
between runs or models are within noise at this sample size.

**FINAL RUN — frozen model, source of truth for the report and `results/mlp/`:**

- Threshold selected from validation set: **0.9996**
- Test performance at this fixed threshold (final reported numbers):
  - Fraud class: precision=0.8116, recall=0.7887, F1=0.80 (support=71)
  - Confusion matrix: 56/71 frauds caught, 13 false positives, 15 missed
  - ROC-AUC: **0.9563**
  - PR-AUC: **0.7479**
  - Accuracy: 0.999342

**These are the headline numbers to use in the report.** Every
exploratory retrain above (initial run, reload, Sept 15 model, and the
from-scratch retrain that was briefly on `main`) is superseded by this
single frozen run — the notebook, `results/mlp/`, and this notes file are
now all consistent with it. Frame it in the report as: "we deliberately
selected the operating threshold on the validation set and evaluated once
on test, avoiding threshold tuning on the test set itself, and froze one
trained model as the source of truth for all reported numbers after
observing run-to-run drift across retrains."

---

## 5. Git / GitHub Workflow Issues (useful for Viva — shows real engineering)

- Created `feature/mlp-model` branch before pushing, per team convention.
- First push attempts failed twice: once due to a broken multi-line shell
  command (long PAT-embedded URL got split across lines), once due to a
  genuinely exposed/leaked PAT (had to revoke and regenerate).
- Second wave of push failures: `403 Permission denied` despite being an
  accepted collaborator. Root cause: **fine-grained PATs cannot target
  repos you don't personally own**, even with full collaborator access —
  GitHub simply won't list them in the repo picker. Confirmed access was
  fine independently (could create branches directly from the GitHub UI).
- **Fix:** switched to a **classic** PAT with the `repo` scope, which
  isn't restricted by repo ownership. Push succeeded immediately after.
- Lesson worth a line in the report/appendix: PAT type matters when
  collaborating on someone else's repo, not just token scope.

---

## 6. Open Items to Resolve With the Team

- [ ] Confirm final 4-model set: repo's "About" currently lists MLP,
      CNN-1D, LSTM, TabNet — but Autoencoder was earlier discussed as a
      possible alternative to TabNet. Needs explicit team confirmation.
- [ ] Agree on a shared `results/` subfolder naming convention before the
      other 3 models start pushing (currently only MLP's outputs exist
      there, so this is the easiest time to set the pattern).
- [ ] Confirm all teammates are computing the same metric set (ROC-AUC,
      PR-AUC, confusion matrix, threshold analysis) in a comparable format
      for the Results & Model Comparison section.

---

## 7. Report Sections This Already Supports

- **Data Preprocessing & Feature Engineering** — mostly covered by §1.
- **Model Architectures** — mostly covered by §2 (needs the "why" written
  in full prose).
- **Experimental Design** — shared splits/seed rationale (§1), fixed
  operating threshold rationale (§4).
- **Critical Analysis & Discussion** — §4 is strong differentiator
  material; §5 is good Viva/engineering-depth material.


## 8. MLP vs. Sequence-Based Models — Expected Differences (for Results & Model Comparison)

The MLP treats each transaction as an independent, unordered feature vector
as it has no concept of transaction order or temporal context beyond
whatever the (already PCA-transformed) V1-V28 features and scaled Time
happen to encode implicitly. This is a reasonable fit for this dataset
since fraud here is framed as a per-transaction classification problem
rather than a sequence-labeling one and it likely explains why a
comparatively simple, fast-to-train MLP already reaches strong ROC-AUC
(0.96+) as there may not be much sequential signal left to exploit once
PCA has already decorrelated the original features.

By contrast, LSTM and CNN-1D are built to exploit ordering and local
temporal patterns for e.g. a customer's spending pattern shifting over a
sequence of transactions, not just a single transaction's own values.
Whether they meaningfully outperform the MLP will depend on:
- Whether the team constructs actual per-customer transaction sequences
  as input (this dataset has no customer ID, so this may not be feasible
  without further preprocessing/assumptions)
- Whether the anonymized/PCA'd nature of V1-V28 has already destroyed the
  kind of raw temporal signal these architectures are designed to exploit

**Prediction to test empirically once those models are done:** the MLP
may prove competitive with or even outperform the sequence models on
this particular dataset, precisely because the data isn't naturally
sequential per-customer — which would itself be a valuable, specific
point for Critical Analysis & Discussion rather than an assumption that
"more complex model = better."
