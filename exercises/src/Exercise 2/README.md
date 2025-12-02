# Exercise 2: Text Classification & Clustering (US vs UK)

This project analyzes parliamentary speeches from the US and UK to determine if they can be distinguished by their vocabulary and language patterns. We apply both unsupervised learning (clustering) and supervised learning (classification) techniques.

## 📂 Project Structure

- `data_cleaner.py`: Preprocesses raw text files (removes metadata, HTML, country names).
- `BM25.py`: Converts cleaned text into numerical vectors using the BM25 algorithm.
- `result.py`: Performs unsupervised clustering (KMeans, DBSCAN, HDBSCAN, GMM) and visualizes results using UMAP.
- `classification.py`: Trains and evaluates 5 supervised classifiers using 10-fold cross-validation.

## 🚀 How to Run

1.  **Clean the Data**:
    ```bash
    python data_cleaner.py
    ```
    *   Input: Raw text files in `data/us/` and `data/uk/`.
    *   Output: Cleaned text in `data/cleaned_us/` and `data/cleaned_uk/`.

2.  **Generate Features (BM25)**:
    ```bash
    python BM25.py
    ```
    *   Output: Sparse matrices (`.npz`) and vocabulary files in `matrices/`.

3.  **Part A: Unsupervised Learning**:
    ```bash
    python result.py
    ```
    *   Output: UMAP visualization plots saved in `images/`.

4.  **Part B: Supervised Learning**:
    ```bash
    python classification.py
    ```
    *   Output: `classification_results.csv` with performance metrics.

---

## 📊 Part A: Analysis of Clustering Results

The `result.py` script generates UMAP visualizations saved in the `images/` folder. Here is how to interpret them:

### UMAP Projection
UMAP (Uniform Manifold Approximation and Projection) reduces the high-dimensional BM25 vectors (thousands of words) into 2 dimensions for visualization. Points that are close together in the plot are similar in terms of their word usage.

### 1. True Labels (`images/umap_true_labels.png`)
- **What it shows**: The ground truth. Red points are US speeches, Blue points are UK speeches.
- **Interpretation**:
    - If you see two distinct, separated blobs (one red, one blue), it means the language used in US and UK parliaments is significantly different and easy to distinguish.
    - If the points are mixed/overlapping, it means the vocabulary is very similar, making the task harder for clustering algorithms.

### 2. Clustering Results
We compare how different unsupervised algorithms group the data without knowing the labels.

- **KMeans (`images/umap_KMeans.png`)**:
    - Forces the data into $k=2$ clusters.
    - **Look for**: Does the split match the True Labels split? KMeans assumes spherical clusters, so it might fail if the natural clusters are irregular shapes.

- **DBSCAN (`images/umap_DBSCAN.png`)**:
    - Density-based clustering. Points in low-density regions are labeled as noise (Cluster -1).
    - **Look for**: How many points are "noise"? Does it find the two main groups, or does it merge them into one big cluster? If `eps` is too small, everything is noise. If too large, everything is one cluster.

- **HDBSCAN (`images/umap_HDBSCAN.png`)**:
    - Hierarchical DBSCAN. It finds clusters of varying densities.
    - **Look for**: Often more robust than DBSCAN. Check if it identifies the two countries as separate dense regions or if it finds sub-topics within the speeches instead.

- **GMM (`images/umap_GMM.png`)**:
    - Gaussian Mixture Models. Probabilistic assignment.
    - **Look for**: Similar to KMeans but more flexible (can handle elliptical clusters). It often works well if the data is somewhat normally distributed in the reduced PCA space.

---

## 📈 Part B: Supervised Learning Results

The `classification.py` script trains five different models to classify the speeches as either US or UK.

### Models Implemented
1.  **Artificial Neural Network (ANN)**: Uses `MLPClassifier`.
2.  **Naive Bayes (NB)**: Uses `MultinomialNB` (standard for text counts).
3.  **Support Vector Machine (SVM)**: Uses `SVC` with a linear kernel.
4.  **Logistic Regression (LoR)**: A strong baseline for text classification.
5.  **Random Forest (RF)**: An ensemble of decision trees.

### Evaluation Methodology
- **10-Fold Cross-Validation**: The data is split into 10 parts. The model is trained on 9 and tested on 1, repeated 10 times.
- **Metrics**: We report the average **Accuracy**, **Precision**, **Recall**, and **F1-Score** across all folds.

Check `classification_results.csv` for the detailed performance numbers.
