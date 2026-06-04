# Machine Learning Model Evaluation Criteria

## 1. Binary Detector Evaluation Criteria

### 1.1 Classification Metrics

| Metric | Target | Description | 
|--------|--------|-------------|
| **Accuracy** | >= 90% | Overall correctness of mangrove vs. non-mangrove classification |
| **Precision ** | >= 88% | Out of all tiles predicted as mangrove, how many are truly mangrove |
| **Recall ** | >= 88% | Out of all actual mangrove tiles, how many were correctly detected |
| **F1-Score** | >= 88% | Combination of precision and recall |
| **AUC-ROC** | >= 0.92 | Area under the Receiver Operating Characteristic curve |

### 1.2 Confidence & Tile Metrics

| Metric | Target | Description | 
|--------|--------|-------------|
| **Prediction Confidence** | — | Confidence score (between and 1) for each prediction | 
| **Probability Distribution** | — | Probability of Non-Mangrove, or Mangrove per prediction |
| **Tile Count** | >= 1 | Number of tiles processed per image | 
| **Batch Processing** | — | Tiles processed in batches (default batch size is 64) | 
| **Confidence Threshold Filter** | configurable | Minimum confidence to include a tile's prediction | 

### 1.3 Tile-Based Processing

| Metric | Target | Description | 
|--------|--------|-------------|
| **Tile Extraction** | 512×512 tiles with 64 pixel overlap | Slices large images into overlapping tiles | 
| **Padded Tiles** | — | Small images are padded to tile size with black borders | 
| **Tile Aggregation** | Mean of per-tile probabilities | Final prediction is the mean of all tile probabilities | 

---

## 2. Multi-class Classifier Evaluation Criteria

We will consider our model as successful if it meets the targets set below in 2.1 and 2.2.

### 2.1 Classification Metrics

| Metric | Target | Description | 
|--------|--------|-------------|
| **Overall Accuracy** | >= 85% | Overall correctness across all 3 mangrove types |
| **Per-Class Precision** | >= 80% each | Precision for Orange, Red, and Yellow individually |
| **Per-Class Recall** | >= 80% each | Recall for Orange, Red, and Yellow individually |
| **Macro F1-Score** | >= 80% | Unweighted mean of per-class F1-scores | 
| **Weighted F1-Score** | >= 80% | F1-score weighted by class support | 

### 2.2 Training Metrics

| Phase | Metric | Target | Description | 
|-------|--------|--------|-------------|
| **Stage 1 (Pretraining)** | Val Accuracy | >= 80% | Validation accuracy after frozen backbone training | 
| **Stage 2 (Fine-tuning)** | Val Accuracy | >= 85% | Validation accuracy after unfreezing last layers | 
| **Early Stopping Trigger** | Patience | 5 epochs | If 5 epochs pass without an improvement to the accuracy, stop testing |
| **Learning Rate Scheduler** | ReduceLROnPlateau | patience=3 | Reduces learning rate when val_loss plateaus | 
| **Final Train-Val Gap** | <= 10% | Difference between train and val accuracy (overfitting indicator) |

---

## 3. Combined Pipeline Evaluation Criteria

### 3.1 Pipeline Metrics

| Metric | Target | Description | 
|--------|--------|-------------|
| **Pipeline Success Rate** | >= 98% | Percentage of uploaded images that produce a full result | 
| **Fallback Handling** | — | Graceful degradation: if one model fails, the other still runs | 
| **Binary → Multi-class Agreement** | — | Multi-class predictions only made when binary model detects mangrove with confidence > 0.5 |

### 3.2 Dataset Quality Metrics

| Metric | Target | Description | 
|--------|--------|-------------|
| **Train/Val Split** | 80/20 | Standard split with a set seed of 42 (reduces randomness) | 
| **Weighted Sampling** | — | Class-weighted sampling to handle imbalance | 
| **Image Corruption Validation** | — | Filters out dark, uniform, and error images | 

---

## 4. Evaluation Procedures

### 4.1 Test Set Evaluation
- Use a held-out test set not seen during training or validation
- Report all metrics from Sections 1–3 on the test set

### 4.2 Error Analysis
- classification_report provides per-class precision, recall, and F1 for error categorization
- A Confusion Matrix provides us with more detail about the models predictions, and can allow us to identify where the model is going wrong.

---
