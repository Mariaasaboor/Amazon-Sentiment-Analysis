# Amazon-Sentiment-Analysis
# Amazon Fine Food Sentiment Analysis: A Comparative Study

## 📌 Project Overview
This project evaluates two distinct approaches to Natural Language Processing (NLP) by classifying the sentiment of Amazon Fine Food reviews. The goal is to compare a traditional "keyword-based" machine learning model against a modern "context-aware" Transformer model.

## 🛠️ Tech Stack
* **Language:** Python 3.13
* **Environment:** Jupyter Notebook / VS Code
* **Libraries:** `pandas`, `scikit-learn`, `transformers` (Hugging Face), `torch`, `matplotlib`, `seaborn`

## 📊 Methodology

### 1. Data Preprocessing
* **Source:** Amazon Fine Food Reviews (Kaggle).
* **Cleaning:** Removed HTML tags, punctuation, and performed case folding using Regular Expressions (Regex).
* **Sampling:** Implemented **Stratified Sampling** to create a balanced dataset of 4,000 reviews (2,000 positive, 2,000 negative) to ensure model fairness.

### 2. Models Compared
* **Baseline Model:** TF-IDF Vectorization + Logistic Regression.
* **Advanced Model:** DistilBERT (Pre-trained on SST-2) using the Hugging Face Pipeline.

---

## 📈 Results & Performance

| Metric              | Baseline (Logistic Reg) | Advanced (DistilBERT) |
| :---                | :---                    | :---                  |
| **Accuracy**        | **83.63%**              | 82.13%                |
| **Negative Recall** | 0.84                    | **0.93**              |
| **Inference Speed** | Very Fast               | Slow (CPU-intensive)  |

### Key Insight: The "Pessimism" of DistilBERT
While the Baseline model achieved higher overall accuracy by focusing on strong keywords (e.g., "delicious," "stale"), DistilBERT demonstrated superior **Recall for negative reviews (0.93)**. This indicates that DistilBERT is highly sensitive to "risk" words and complex sentence structures, making it safer for identifying customer complaints despite the lower accuracy score.



---

## Error Analysis
Through a "Disagreement Analysis," I identified specific cases where the models diverged. For instance, in reviews discussing "risk" or "controversy," DistilBERT often predicted a negative sentiment due to its deep contextual understanding, whereas the Baseline correctly identified the overall positive sentiment by anchoring on specific positive nouns.

## How to Use
1. Clone the repository.
2. Ensure you have the `Reviews.csv` file in the `/Data` directory.
3. Install dependencies: `pip install pandas scikit-learn transformers torch matplotlib seaborn`
4. Run `assignment.ipynb`.
