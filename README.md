你可以用以下结构来编写你的 GitHub 项目 `README.md`，完整覆盖你们团队在 AIDI1003 项目中所做的 Reddit 加拿大移民文本分类研究。以下是根据你提供的项目内容量身定制的模板：

---

# 🇨🇦 Reddit Immigration Topic Classifier

This project analyzes Reddit discussions on Canadian immigration to detect topic shifts before and after a major policy reform on **May 31, 2023**.

## Project Overview

We built a **text classification pipeline** to categorize Reddit posts into immigration-related topics using both traditional machine learning and transformer models. Our main goal was to identify whether discussions on **Express Entry** increased after a new category-based invitation system was introduced.

## Data Disclaimer

The `/data/` directory is intentionally excluded from this repository due to size and privacy considerations.

All Jupyter notebooks are provided and demonstrate the full data processing and model training pipeline. To reproduce results:
- Use your own Reddit dataset in the same structure (title, body, date)
- Follow steps in `1_Data_Cleaning.ipynb` to preprocess

> For educational or academic access to our dataset, please contact us.

## Project Steps

### 1. Data Collection

* Source: `.zst` Reddit dumps from **Pushshift** (Mar–Aug 2023).
* Subreddits: `r/ImmigrationCanada`, `r/CanadaImmigrant`.
* Total posts: **10,664**.
* Focus: Submissions only (not comments) for richer context.

### 2. Data Cleaning

* Merged title and body into a single field.
* Cleaned using `unidecode`, removed stopwords, HTML, punctuation, etc.
* Removed deleted/empty posts.
* Final CSV contains cleaned text.

### 3. Data Labeling & Sampling

* Defined 8 categories:

  * Express Entry
  * Family Sponsorship
  * PGWP
  * PNP
  * Refugee
  * Student Permit
  * Work Permit
  * Other

* Used **keyword-based labeling script** for initial tagging.

* Manually reviewed **\~1,090** posts (10% sample), ensuring class balance.

### 4. Modeling

* **Baseline Models**: Logistic Regression, SVM using TF-IDF.
* **Deep Models**: Fine-tuned `DistilBERT` and `RoBERTa-base` via Hugging Face.
* Best model: **DistilBERT**

  * Accuracy: **78.6%**
  * Macro F1: **0.79**
  * Express Entry F1: **0.69** (after targeted augmentation)

### 5. Inference

* Used final `DistilBERT` model to predict labels for unlabeled posts.
* Saved predicted topics in new CSV.

### 6. Policy Analysis

* Added a `policy_period` flag (before/after May 31).
* Visualized monthly topic trends.
* Found a **clear spike in Express Entry posts** after the policy change.

## Key Result

> Express Entry discussions rose significantly after May 31st, especially peaking in July — suggesting the policy reform had real impact on public discourse.

## Tech Stack

* Python
* pandas, scikit-learn
* Hugging Face Transformers
* matplotlib / seaborn
* Jupyter / Google Colab

## Project Structure

```
/data              # Raw and cleaned datasets
/scripts           # Python scripts for data prep and modeling
/models            # Saved model checkpoints
/notebooks         # Colab notebooks
/output            # Inference results and visualizations
README.md
```

## 📎 References

* [Pushshift Reddit Dump](https://academictorrents.com/details/36f5241a49e676949b26f1395d05e4e02aa1ecf9)
* [Policy Link - Canada.ca](https://www.canada.ca/en/immigration-refugees-citizenship/news/2023/05/canada-launches-new-process-to-welcome-skilled-newcomers.html)

