SE4050 – Credit Card Fraud Detection (Deep Learning Assignment)

Group ID: SE4050_G39

1. Project Title

Credit Card Fraud Detection using Deep Learning — a comparative study of four architectures (MLP, TabNet, LSTM, CNN-1D) on a severely imbalanced, real-world transaction dataset.

2. Problem Statement

Credit-card fraud is a high-impact, low-frequency event: fraudulent transactions make up only a tiny fraction of all transactions, which makes naïve accuracy a misleading metric and requires class-imbalance-aware training plus threshold-based (rather than default-0.5) decision rules. This project implements and compares four deep-learning architectures — MLP, TabNet, LSTM, and CNN-1D — on the same shared dataset splits, evaluating them with precision/recall/F1/ROC-AUC/PR-AUC at a validation-selected operating threshold, and critically discusses each architecture's suitability for this tabular, non-sequential fraud-detection task.

3. Dataset
Name: Credit Card Fraud Detection Dataset
Source / Link: Kaggle — mlg-ulb/creditcardfraud (originally released by Worldline and the Machine Learning Group, Université Libre de Bruxelles)
Citation: Machine Learning Group – ULB. Credit Card Fraud Detection. Kaggle, Worldline & ULB collaboration.
Size: 284,807 transactions recorded over two days in September 2013 by European cardholders; 283,726 rows remain after removing 1,081 duplicate rows during preprocessing (see notebooks/01_preprocessing.py).
Classes: 0 = Legitimate, 1 = Fraudulent — 473 of 283,726 transactions are fraudulent after deduplication (≈0.167%), a severe class imbalance.
Features: 30 numeric input features — V1–V28 are PCA-transformed components (protecting cardholder confidentiality), plus Time (seconds elapsed since the first transaction) and Amount (transaction value, log-transformed to Amount_log before use).
4. Models Implemented
MLP (Multi-Layer Perceptron) — Bowatte W.M.E (IT22001498)
CNN-1D (1D Convolutional Neural Network) — De Silva M.H.S.A (IT21178290)
LSTM (Long Short-Term Memory) — Bandara R.M.D.N (IT22578150)
TabNet (attention-based tabular model) — Manamendra J.N (IT22608772)
5. Repository Structure
/notebooks       → Google Colab notebooks (MLP.ipynb, CNN1D.ipynb, LSTM.ipynb, TabNet.ipynb)
                    + 01_preprocessing.py (shared preprocessing script, run once)
/results         → saved plots, confusion matrices, metrics CSVs (per model)
/configs         → saved hyperparameters/configs per model (JSON, includes random seed)
README.md
requirements.txt
.gitignore
Members.txt
6. Setup Instructions
Requirements
Python 3.9+
See requirements.txt for exact package versions (pandas, numpy, scikit-learn, tensorflow, torch, pytorch-tabnet, matplotlib, seaborn).
A Kaggle account (free) to download the dataset.
Recommended: Google Colab (GPU runtime) for training MLP/TabNet/LSTM/CNN-1D — the notebooks were originally developed there.
Installation
bash
git clone https://github.com/manamendraJN/SE4050-Fraud-Detection.git
cd SE4050-Fraud-Detection
pip install -r requirements.txt
7. How to Run
Dataset Access
Download creditcard.csv from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud (requires a free Kaggle login; download via the website or the Kaggle API).
Place the file at data/creditcard.csv relative to the repository root (this is the path 01_preprocessing.py expects — adjust DATA_PATH at the top of the script if you place it elsewhere).
Running the Notebooks
Run the shared preprocessing script once (do not re-run it per model — every model must train/validate/test on the identical splits so results stay comparable):
bash
   python notebooks/01_preprocessing.py

This cleans the data (drops 1,081 duplicate rows), log-transforms Amount, performs a 70/15/15 stratified split (random_state=42, preserving the ~0.167% fraud ratio in every split), fits a StandardScaler on Time/Amount_log using the training partition only, and writes train.csv, val.csv, test.csv to data/processed/. 2. Open each model notebook (MLP.ipynb, TabNet.ipynb, LSTM.ipynb, CNN1D.ipynb) in Jupyter or Google Colab and run all cells top to bottom. Each notebook loads the same three CSVs from data/processed/ — do not point different models at different data. 3. Each notebook saves its trained model, metrics CSV, and result figures under results/<model_name>/.

8. Reproducibility
Random Seed: 42, fixed across preprocessing (data split, shuffling), and training (where the framework supports it), for every model.
Saved Configurations: Each model's hyperparameters, architecture, and seed are saved under configs/ as JSON (mlp_config.json, tabnet_config.json, lstm_config.json, cnn1d_config.json) so results can be reproduced without re-tuning.
9. Results Summary

All models are evaluated on the identical held-out test split (42,559 transactions, 71 fraudulent) at a validation-F1-maximising operating threshold, except LSTM (see note below).

Model	Accuracy	Precision	Recall	F1	ROC-AUC	PR-AUC	Threshold
MLP	0.9993	0.8116	0.7887	0.80	0.9563	0.7479	0.9996
TabNet	0.9993	0.8182	0.7606	0.7883	0.9500	0.7204	0.385
CNN-1D	0.9993	0.8209	0.7746	0.7971	0.9546	0.7428	0.845
LSTM*	0.9902	0.66	0.8919	0.7586	0.9923	0.8986	0.89

*LSTM caveat: the current LSTM (1).ipynb run evaluates on a synthetic fallback dataset (28,492 rows, SMOTE-balanced) rather than the shared train.csv/val.csv/test.csv splits above — its data-loading cell needs to be pointed at the shared processed CSVs and re-run before its numbers are directly comparable to MLP/TabNet/CNN-1D. Reported here for completeness, not as a like-for-like comparison.

10. Team Contributions
Student ID	Name	Model Owned	Contribution
IT22001498	Bowatte W.M.E	MLP	MLP architecture, training, evaluation, and report sections
IT22608772	Manamendra J.N	TabNet	TabNet architecture, tuning, training, evaluation, and report sections
IT22578150	Bandara R.M.D.N	LSTM	LSTM architecture, training, evaluation, and report sections
IT21178290	De Silva M.H.S.A	CNN-1D	CNN-1D architecture, training, evaluation, and report sections

All members contributed to the shared preprocessing pipeline (01_preprocessing.py), cross-model results comparison, and the critical analysis and discussion section of the report.

11. References
Machine Learning Group – ULB. Credit Card Fraud Detection Dataset. Kaggle. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Arik, S. Ö. and Pfister, T. (2020). TabNet: Attentive Interpretable Tabular Learning. arXiv:1908.07442.
DreamQuark. pytorch-tabnet (Python library), used for the TabNet implementation.
Abadi, M. et al. TensorFlow: Large-scale machine learning on heterogeneous systems, used for the MLP and LSTM implementations.
Paszke, A. et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library, used for the TabNet and CNN-1D implementations.
Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
Project Description

"Deep learning models (MLP, CNN-1D, LSTM, TabNet) for credit card fraud detection on the Kaggle MLG-ULB dataset — SE4050 Deep Learning assignment."
