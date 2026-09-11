# CSE715: Multimodal Music Understanding with Graph Neural Networks and BERT

This repository contains the research codebase, experimental pipelines, visualizations, and report for **CSE715: Multimodal Music Understanding with Graph Neural Networks and BERT**. The project explores how natural-language textual context, audio representations, and graph-structured acoustic modeling interact across classification, multimodal fusion, and cross-modal retrieval.

---

## Collaborators

* **Israt Moyeen Noumi**
  ID- 1000060098
  Department of Computer Science and Engineering, BRAC University  
  Email: israt.moyeen.noumi@g.bracu.ac.bd

* **Mysha Maliha Annisa**
  ID- 21101243
  Department of Computer Science and Engineering, BRAC University  
  Email: mysha.maliha.annisa@g.bracu.ac.bd

* **Tazkera Sattar**
  ID- 21101282
  Department of Computer Science and Engineering, BRAC University  
  Email: tazkera.sattar@g.bracu.ac.bd

---

## Project Overview & Tasks

The project is structured into four core research tasks, implemented in end-to-end Jupyter notebooks:

### Task 1: Caption-to-Tag Multi-Label Classification on MusicCaps
* **Notebook**: [notebooks/task-1.ipynb](notebooks/task-1.ipynb)
* **Dataset**: Google MusicCaps (5,521 natural-language music descriptions paired with independent aspect tags)
* **Architecture**: Pretrained BERT (`bert-base-uncased`) encoder with a linear multi-label classification head and BCEWithLogitsLoss.
* **Objective**: Evaluate whether free-text human captions alone provide sufficient contextual semantic signal to recover top-50 aspect tags (genres, instruments, moods, tempos) without access to raw audio waveforms.
* **Key Visualizations**: [plots/task1_training_history.png](plots/task1_training_history.png)

---

### Task 2: GNN vs. CNN Single-Label Genre Classification on FMA-Small
* **Notebook**: [notebooks/task-2.ipynb](notebooks/task-2.ipynb)
* **Dataset**: Free Music Archive Small (FMA-small: 8,000 tracks, 8 balanced genres)
* **Architectures**:
  * **Graph Neural Network (GNN)**: Audio clips are partitioned into 3-second temporal segments. Each segment node extracts multi-statistic acoustic features (MFCC, chroma, spectral contrast, tonnetz). Nodes are connected via temporal sequence edges and k-NN acoustic similarity edges, processed through a multi-layer GraphSAGE / GAT architecture with global mean pooling.
  * **Convolutional Neural Network (CNN) Baseline**: 4-block 2D CNN operating over 128-band mel-spectrogram representations.
* **Objective**: Compare coarse segment-level graph representations against fine-grained time-frequency spectrogram representations under identical split protocols and parameter budgets.
* **Key Visualizations**: [plots/task2_gnn_training_history.png](plots/task2_gnn_training_history.png), [plots/task2_gnn_vs_cnn.png](plots/task2_gnn_vs_cnn.png)

---

### Task 3: Multimodal GNN–BERT Fusion for Multi-Context Music Understanding
* **Notebook**: [notebooks/task-3.ipynb](notebooks/task-3.ipynb)
* **Dataset**: Free Music Archive Medium (FMA-medium: top-10 genres, artist-disjoint split)
* **Architectures & 4-Way Ablation**:
  * **Model A (BERT-only)**: DistilBERT encoder processing rich textual metadata (track title, artist background, production arrangement, engagement stats) with strict anti-leakage genre-keyword masking.
  * **Model B (GNN-only)**: GraphSAGE network over temporal audio-segment graphs.
  * **Model C (Early Concatenation Fusion)**: Concatenation of pooled audio-graph embedding and textual [CLS] embedding ($z = [g \,;\, t]$).
  * **Model D (Cross-Attention Fusion)**: Multi-head cross-attention where the audio-graph embedding queries token-level hidden states from BERT ($Q = g W_Q, K = H_{text} W_K, V = H_{text} W_V$).
* **Objective**: Investigate whether multimodal fusion yields statistically meaningful gains over single-modality baselines under strict anti-leakage controls.
* **Key Visualizations**: [plots/task3_ablation_performance.png](plots/task3_ablation_performance.png), [plots/task3_tsne.png](plots/task3_tsne.png)

---

### Task 4: Cross-Modal Audio–Text Retrieval & Zero-Shot Tagging via Contrastive Learning
* **Notebook**: [notebooks/task-4.ipynb](notebooks/task-4.ipynb)
* **Dataset**: MusicCaps audio clips matched with human captions
* **Architecture**: Symmetric dual-encoder (CLIP-style) framework:
  * **Audio Branch**: AudioGraphEncoder (segmentation into mel, MFCC, and chroma features $\to$ GraphSAGE $\to$ projection head $\to$ $L_2$ normalization).
  * **Text Branch**: TextBERTEncoder (`bert-base-uncased` $\to$ projection head $\to$ $L_2$ normalization).
* **Training & Loss**: Optimized using bidirectional symmetric InfoNCE contrastive loss over shared embedding space.
* **Objective**: Enable zero-shot music tagging and bidirectional cross-modal retrieval:
  * Caption-to-Audio retrieval ($R@1, R@5, R@10$)
  * Audio-to-Caption retrieval ($R@1, R@5, R@10$)
  * Zero-shot classification via prompt-based tag matching.

---

## Repository Structure

```text
CSE715-Project/
├── notebooks/
│   ├── task-1.ipynb          # Task 1: MusicCaps BERT caption-to-tag classification
│   ├── task-2.ipynb          # Task 2: FMA-small GNN vs. CNN genre classification
│   ├── task-3.ipynb          # Task 3: FMA-medium GNN-BERT multimodal ablation study
│   └── task-4.ipynb          # Task 4: MusicCaps dual-encoder contrastive learning & retrieval
├── plots/
│   ├── task1_training_history.png
│   ├── task2_gnn_training_history.png
│   ├── task2_gnn_vs_cnn.png
│   ├── task3_ablation_performance.png
│   └── task3_tsne.png
├── report/
│   └── CSE715_Report.pdf     # Full research paper and experimental results
└── README.md                 # Project documentation and summary
```

---

## Environment & Requirements

The experiments utilize PyTorch and can be run locally or on Kaggle environments with GPU acceleration:

* Python 3.10+
* PyTorch & Torchaudio
* PyTorch Geometric (`torch-geometric`)
* Hugging Face Transformers (`transformers`, `datasets`)
* Librosa & SoundFile
* Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn

---

## Research Report

The complete academic manuscript detailing the problem formulation, methodology, experimental setup, ablation findings, and discussion is available in [report/CSE715_Report.pdf](report/CSE715_Report.pdf).
