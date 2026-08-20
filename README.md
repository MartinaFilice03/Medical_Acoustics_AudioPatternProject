# Patient-Independent Medical Acoustics Classification Pipeline

This project implements a comprehensive machine learning and data mining pipeline for the automated classification of biological and medical auscultation recordings (cardiopulmonary sounds) using the HLS-CMDS repository. It benchmarks traditional handcrafted acoustic features, data-driven dimensionality reduction, optimized supervised classifiers, sequence modeling, and deep learning embeddings while enforcing strict methodological safeguards against cross-subject information leakage.

***

## Dataset Structure

As a dataset, I utilized the **HLS-CMDS** repository (Heart and Lung Sounds Dataset Recorded from a Clinical Manikin using Digital Stethoscope), which contains uncompressed `.wav` audio recordings sampled at high acoustic fidelity and detailed clinical metadata. Specifically, the CSV files and audio folders saved within the project directory consist of:
* **`Mix.csv`** and the **`Mix`** folder: contain records and uncompressed audio tracks representing combined cardiopulmonary auscultation mixtures.
* **`HS.csv`** and the **`HS`** folder: contain records and audio tracks isolating pure cardiac cycles.
* **`LS.csv`** and the **`LS`** folder: contain records and audio tracks isolating pulmonary respiratory phases.

***

## Project Structure

```text
├── Medical_Acoustics.ipynb   # Interactive Jupyter Notebook containing the full pipeline
├── medical_acoustics.py      # Python script version of the computational pipeline
└── README.md                 # Project documentation

***

## Project Features

The pipeline is structured into the following stages:

* **Environment Setup & Dataset Ingestion**: Connects to Google Drive and structures the root directories for the HLS-CMDS medical acoustics collection, mapping metadata files (`Mix.csv`, `HS.csv`, `LS.csv`) directly to uncompressed `.wav` audio directories (`Mix`, `HS`, `LS`).
* **Patient-Independent Partitioning**: Implements a custom splitting routine using `GroupShuffleSplit` based on unique patient identifiers, strictly prohibiting subject-level overlap and establishing robust generalization boundaries between training and testing cohorts.
* **Handcrafted Feature Extraction**: Extracts **229 handcrafted features** using `librosa` (encompassing Root Mean Square energy, Zero-Crossing Rate, advanced spectral descriptors, normalized spectral entropy, and 13 MFCCs alongside $\Delta$, $\Delta^2$ delta derivatives) with mid-term statistical aggregation and near-zero variance pruning.
* **Leak-Free Preprocessing & PCA Data Mining**: Applies median imputation (`SimpleImputer`) and z-score standardization (`StandardScaler`) learned exclusively on the training partition, followed by Principal Component Analysis (PCA) to project data onto an optimized latent subspace targeting a cumulative explained variance threshold of 95%.
* **Supervised Classification & Tuning**: Benchmarks a diverse suite of supervised algorithms—Support Vector Machines (Linear and RBF kernels), $k$-NN, Random Forest, and Gradient Boosting—optimized via `GridSearchCV` driven by weighted F1-score (`f1_weighted`) combined with `GroupKFold` cross-validation.
* **Sequential Stochastic Modeling**: Integrates class-balanced SVM optimization (`class_weight='balanced'`) to mitigate majority class bias and trains independent Gaussian Hidden Markov Models (`hmm.GaussianHMM`) on frame-level sequence segments evaluated via maximum log-likelihood.
* **Deep Learning for Embedding Space**: Converts audio tracks into 64-band Log-Mel Spectrograms and trains a deep Convolutional Neural Network (CNN) in TensorFlow/Keras featuring a 64-dimensional latent embedding layer.
* **Content-Based Audio Retrieval**: Implements an audio spotting retrieval module utilizing Cosine Similarity on deep latent embeddings to identify acoustically similar pathological profiles.
* **Model Robustness & Explainable AI (XAI)**: Executes multi-seed stability evaluations (`[7, 19, 42, 73, 101]`) to verify deterministic model immunity and computes Permutation Feature Importance to isolate the top principal components driving clinical decisions.

***

## Requirements and Dependencies

The core Python libraries required to run this project include:

* **Audio Processing**: `librosa`
* **Machine Learning**: `scikit-learn`, `hmmlearn`
* **Deep Learning**: `TensorFlow` / `Keras`
* **Data Science & Visualization**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `tqdm`
