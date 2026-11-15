# Student Academic Performance Prediction (Deep Learning)

A Deep Neural Network that predicts a student's final academic performance category from demographic,
academic, social and behavioural information, compared against three traditional machine-learning models
and served through a Streamlit web app.

## Dataset

UCI Student Performance dataset, Portuguese course (`data/student-por.csv`): 649 students, 32 input
attributes and the final grade G3 (0-20). There are no missing values and no duplicate rows.

The final grade is converted into three performance categories:

| Category | Final grade G3 | Students | Share |
|---|---|---|---|
| Low | 0-9 (fail) | 100 | 15.4% |
| Medium | 10-13 (pass) | 355 | 54.7% |
| High | 14-20 (good / excellent) | 194 | 29.9% |

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

python eda.py                     # exploratory data analysis -> outputs/
python train.py                   # feature selection, training, tuning, evaluation (about 6 minutes on CPU)
streamlit run app.py              # web app at http://localhost:8501
```

`python train.py --trials 4` runs a shorter hyperparameter search.

## Project structure

| File | Purpose |
|---|---|
| `common.py` | Shared settings: paths, grade-to-category rule, feature descriptions, chart style |
| `eda.py` | Exploratory data analysis: text summary and six figures |
| `train.py` | Preprocessing, feature selection, traditional ML, DNN tuning, final training, evaluation |
| `app.py` | Streamlit app: prediction form, model performance, data exploration |
| `data/` | The dataset and its attribute description |
| `models/` | Trained DNN, fitted preprocessor, traditional models, `metadata.json` |
| `outputs/` | Figures, comparison table, tuning results, feature ranking, training log |

## Method

1. **Preprocessing.** Stratified 80/20 split (519 training, 130 test students). Numeric and ordinal
   attributes are standardised; nominal attributes are one-hot encoded. The preprocessor is fitted on
   training data only.
2. **Exploratory data analysis.** The earlier period grades are by far the strongest correlates of G3
   (G2: 0.92, G1: 0.83), followed by past failures (-0.39), study time (0.25) and mother's education (0.24).
   See `outputs/eda_summary.txt` and figures 01-06.
3. **Feature selection.** All 32 attributes are ranked by mutual information and by random-forest
   importance on the training set, and the 15 with the best average rank are kept: G2, G1, failures, Medu,
   higher, age, school, Walc, Fedu, Mjob, absences, freetime, studytime, health and reason.
4. **Traditional ML.** Logistic Regression, Random Forest and SVM (RBF), each tuned by grid search with
   5-fold cross-validation on macro F1.
5. **DNN design and tuning.** Fully connected network with ReLU hidden layers, dropout, L2 regularisation
   and a 3-unit softmax output, trained with Adam, class weights, early stopping and learning-rate
   reduction. A random search of 16 configurations (layer sizes, dropout, L2, learning rate, batch size,
   batch normalisation) is scored with 3-fold cross-validation.
6. **Evaluation.** Every model is scored once on the held-out test set.

**Tuned DNN:** 15 features (22 inputs after encoding) → Dense 64 → Dense 32 → Softmax 3, dropout 0.1,
L2 0.001, learning rate 0.0003, batch size 32, no batch normalisation; best epoch 85.

## Results (130 held-out test students)

| Model | CV macro F1 | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC |
|---|---|---|---|---|---|---|
| Random Forest | 0.819 | 0.869 | 0.847 | 0.829 | 0.836 | 0.963 |
| SVM (RBF) | 0.817 | 0.862 | 0.840 | 0.824 | 0.830 | 0.967 |
| Logistic Regression | 0.823 | 0.854 | 0.844 | 0.815 | 0.826 | 0.953 |
| **Deep Neural Network** | 0.785 | 0.831 | 0.794 | 0.829 | 0.808 | 0.955 |

DNN per-class results:

| Category | Precision | Recall | F1 | Test students |
|---|---|---|---|---|
| Low | 0.652 | 0.750 | 0.698 | 20 |
| Medium | 0.889 | 0.789 | 0.836 | 71 |
| High | 0.841 | 0.949 | 0.892 | 39 |

### Discussion

- The DNN reaches 83.1% accuracy but does not beat the traditional models. With only 519 training rows of
  tabular data this is expected: tree ensembles and linear models need less data than neural networks.
- The differences are small. The gap between the DNN and Random Forest is five students out of 130, so the
  ranking of the four models should not be treated as conclusive.
- The DNN identifies the most Low-performing students (15 of 20, against 13 for each traditional model),
  at the cost of more Medium students flagged as Low. That trade-off suits an early-warning use.
- No model ever confuses Low with High; all errors are between neighbouring categories.

## Limitations

- **G1 and G2 are used as inputs.** They are earlier grades in the same course and drive most of the
  accuracy. The model is therefore useful after the second period, not at the start of the year.
- **The employability grade is a label mapping, not a measured outcome.** The dataset contains no
  employment data. The app maps Low, Medium and High to "Needs significant support", "Employable with
  upskilling" and "Highly employable".
- The data covers two Portuguese secondary schools (2005-2006), so the model may not transfer to other
  populations.
- TensorFlow runs on CPU on native Windows; results are reproducible with the fixed seed (42).

## Reference

P. Cortez and A. Silva, "Using Data Mining to Predict Secondary School Student Performance", 2008.
Dataset: https://archive.ics.uci.edu/dataset/320/student+performance
