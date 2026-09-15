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
- Split sizes: Train (198,608, 32), Val (42,559, 32), Test (42,559, 32).

**Why it matters for the report:** fair cross-model comparison depends on
every teammate using these exact splits — worth stating explicitly in
Experimental Design as a controlled variable.

---

## 2. MLP Architecture — decisions and rationale

```
Dense(64, ReLU) → BatchNorm → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)
Optimizer: Adam, lr=0.001
Loss: binary cross-entropy
Class weights: sklearn compute_class_weight("balanced") — handles ~0.17% fraud imbalance
Early stopping: monitor=val_auc, mode=max, patience=8, restore_best_weights=True
```

Things to justify specifically in the report rather than leaving generic:
- Why 64→32 (capacity vs. overfitting risk on a fairly small effective
  positive-class sample)
- Why dropout 0.3 at both layers
- Why early stopping on **val_auc** specifically rather than val_loss —
  AUC is threshold-independent and more meaningful under heavy imbalance
- Why class weighting was chosen over resampling (e.g. SMOTE) — didn't
  test SMOTE, worth naming as a limitation/future work item

---

## 3. Initial Training Results

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
identical re-runs** of the same trained model. This is not primarily a
float-saturation artifact (the saturation cluster exists but isn't what's
driving the instability) — it's a **sample-size effect**: the test set
only contains **71 fraud cases**, so the F1-vs-threshold curve is being
optimized over a very small, noisy set of positives. Small floating-point
differences between runs (GPU non-determinism) are enough to shift which
exact threshold value "wins" the argmax.

**Methodological fix adopted:** stop selecting the operating threshold
from the **test set** (this is a mild form of test-set leakage — tuning a
decision using the same data used to report final performance). Instead:
select the threshold using the **validation set**, then apply that fixed
threshold once to the test set and report those numbers as final.

```python
val_proba = model.predict(X_val).ravel()
val_precisions, val_recalls, val_thresholds = precision_recall_curve(y_val, val_proba)
val_f1 = 2 * (val_precisions * val_recalls) / (val_precisions + val_recalls + 1e-9)
best_val_idx = np.argmax(val_f1)
chosen_threshold = val_thresholds[best_val_idx]

y_pred_final = (y_proba >= chosen_threshold).astype(int)
# report classification_report(y_test, y_pred_final, ...) using this fixed threshold
```

**Why this is good report material:** it's a specific, defensible insight
that most groups working on this exact dataset likely won't catch or
discuss (per the differentiation goal vs. the earlier report reviewed).
Write it up as: what we observed → why it happens → what we changed → why
that's more methodologically sound. This is exactly the kind of reasoning
the Critical Analysis section (30% of grade) rewards.

**Status:** validation-based threshold selection code written; final
numbers from this approach not yet recorded — **update this section once
run**.

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
