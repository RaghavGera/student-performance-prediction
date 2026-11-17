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
| `app.py` | Streamlit app: prediction form, model performance |
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
3. **Feature selection (two stages, training data only).**
   - *Stage 1, filter:* all 32 attributes are ranked by mutual information and by random-forest importance,
     and the 15 with the best average rank become candidates.
   - *Stage 2, DNN permutation importance:* with 3-fold cross-validation on the training set, a DNN is
     trained on two folds and each candidate is shuffled in turn in the third fold. A candidate is kept only
     if shuffling it lowers macro F1 by more than 0.002, i.e. only if the model actually uses it.
     This removed health, Medu, studytime and higher, whose shuffling did not hurt (or slightly helped).
   - *Final 11 features:* G2, G1, failures, Fedu, freetime, age, reason, absences, school, Walc, Mjob.
     By group: demographic (school, age, Fedu, Mjob), academic (reason, failures, absences, G1, G2),
     social and behavioural (freetime, Walc).
4. **Traditional ML.** Logistic Regression, Random Forest and SVM (RBF), each tuned by grid search with
   5-fold cross-validation on macro F1.
5. **DNN design and tuning.** Fully connected network with ReLU hidden layers, dropout, L2 regularisation
   and a 3-unit softmax output, trained with Adam, class weights, early stopping and learning-rate
   reduction. A random search of 16 configurations (layer sizes, dropout, L2, learning rate, batch size,
   batch normalisation) is scored with 3-fold cross-validation.
6. **Evaluation.** Every model is scored once on the held-out test set.

**Tuned DNN:** 11 features (18 inputs after encoding) → Dense 128 → Dense 64 → Softmax 3 (10,883 trainable
parameters), dropout 0.3, L2 0.01, learning rate 0.001, batch size 16, no batch normalisation; trained 90
epochs, best weights from epoch 65.

## Results (130 held-out test students)

| Model | CV macro F1 | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC |
|---|---|---|---|---|---|---|
| SVM (RBF) | 0.822 | 0.885 | 0.869 | 0.854 | 0.859 | 0.974 |
| Random Forest | 0.835 | 0.854 | 0.834 | 0.831 | 0.831 | 0.969 |
| Logistic Regression | 0.828 | 0.838 | 0.820 | 0.806 | 0.811 | 0.960 |
| **Deep Neural Network** | 0.801 | 0.838 | 0.808 | 0.818 | 0.808 | 0.968 |

DNN per-class results:

| Category | Precision | Recall | F1 | Test students |
|---|---|---|---|---|
| Low | 0.722 | 0.650 | 0.684 | 20 |
| Medium | 0.891 | 0.803 | 0.844 | 71 |
| High | 0.813 | 1.000 | 0.897 | 39 |

### Discussion

- The DNN reaches 83.8% accuracy (109 of 130), level with Logistic Regression and behind SVM (88.5%) and
  Random Forest (85.4%). With only 519 training rows of tabular data this is expected: SVMs, tree ensembles
  and linear models need less data than neural networks.
- The differences are small and unstable. Before stage-2 pruning (15 features) Random Forest was best at
  86.9% and SVM scored 86.2%; with 11 features SVM is best. One test student is worth 0.77 percentage
  points, so the ranking of the four models should not be treated as conclusive.
- Pruning from 15 to 11 features kept DNN performance the same (macro F1 0.808 both times, accuracy 83.1%
  → 83.8%, ROC-AUC 0.955 → 0.968): the four removed features carried no useful information for the model.
- The DNN recognises every High student (39 of 39) and 13 of 20 Low students.
- No model ever confuses Low with High; all errors are between neighbouring categories.
- Only two social and behavioural attributes survive selection (free time and weekend alcohol use), and
  their effect is small. Their influence on performance is largely already reflected in G1 and G2.

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
