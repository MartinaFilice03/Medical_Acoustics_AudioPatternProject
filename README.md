# Patient-Independent Medical Acoustics Classification Pipeline

This project implements a comprehensive machine learning and data mining pipeline for the automated classification of biological and medical auscultation recordings (cardiopulmonary sounds) using the HLS-CMDS repository[cite: 2]. It benchmarks traditional handcrafted acoustic features, data-driven dimensionality reduction, optimized supervised classifiers, sequence modeling, and deep learning embeddings while enforcing strict methodological safeguards against cross-subject information leakage[cite: 2].

***

## Project Features

The pipeline is structured into the following stages:

* **Environment Setup & Dataset Ingestion**: Connects to Google Drive and structures the root directories for the HLS-CMDS medical acoustics collection, mapping metadata files (`Mix.csv`, `HS.csv`, `LS.csv`) directly to uncompressed `.wav` audio directories (`Mix`, `HS`, `LS`)[cite: 2].
* **Patient-Independent Partitioning**: Implements a custom splitting routine using `GroupShuffleSplit` based on unique patient identifiers, strictly prohibiting subject-level overlap and establishing robust generalization boundaries between training and testing cohorts[cite: 2].
* **Handcrafted Feature Extraction**: Extracts **229 handcrafted features** using `librosa` (encompassing Root Mean Square energy, Zero-Crossing Rate, advanced spectral descriptors, normalized spectral entropy, and 13 MFCCs alongside $\Delta$, $\Delta^2$ delta derivatives) with mid-term statistical aggregation and near-zero variance pruning[cite: 2].
* **Leak-Free Preprocessing & PCA Data Mining**: Applies median imputation (`SimpleImputer`) and z-score standardization (`StandardScaler`) learned exclusively on the training partition, followed by Principal Component Analysis (PCA) to project data onto an optimized latent subspace targeting a cumulative explained variance threshold of 95%[cite: 2].
* **Supervised Classification & Tuning**: Benchmarks a diverse suite of supervised algorithms—Support Vector Machines (Linear and RBF kernels), $k$-NN, Random Forest, and Gradient Boosting—optimized via `GridSearchCV` driven by weighted F1-score (`f1_weighted`) combined with `GroupKFold` cross-validation[cite: 2].
* **Sequential Stochastic Modeling**: Integrates class-balanced SVM optimization (`class_weight='balanced'`) to mitigate majority class bias and trains independent Gaussian Hidden Markov Models (`hmm.GaussianHMM`) on frame-level sequence segments evaluated via maximum log-likelihood[cite: 2].
* **Deep Learning for Embedding Space**: Converts audio tracks into 64-band Log-Mel Spectrograms and trains a deep Convolutional Neural Network (CNN) in TensorFlow/Keras featuring a 64-dimensional latent embedding layer[cite: 2].
* **Content-Based Audio Retrieval**: Implements an audio spotting retrieval module utilizing Cosine Similarity on deep latent embeddings to identify acoustically similar pathological profiles[cite: 2].
* **Model Robustness & Explainable AI (XAI)**: Executes multi-seed stability evaluations (`[7, 19, 42, 73, 101]`) to verify deterministic model immunity and computes Permutation Feature Importance to isolate the top principal components driving clinical decisions[cite: 2].

***

## Requirements and Dependencies

The core Python libraries required to run this project include:

* **Audio Processing**: `librosa`[cite: 2]
* **Machine Learning**: `scikit-learn`, `hmmlearn`[cite: 2]
* **Deep Learning**: `TensorFlow` / `Keras`[cite: 2]
* **Data Science & Visualization**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `tqdm`[cite: 2]
