"""Train and evaluate the Student Academic Performance prediction models.

Pipeline
  1. Preprocessing        - target binning, stratified train/test split, scaling + one-hot encoding
  2. Feature selection    - mutual information + random-forest importance (training data only)
  3. Traditional ML       - Logistic Regression, Random Forest, SVM (grid search, 5-fold CV)
  4. DNN tuning           - random search over architecture / regularisation / learning rate (3-fold CV)
  5. Final DNN            - retrained with the best hyperparameters, early stopping
  6. Evaluation           - held-out test set: accuracy, precision, recall, F1, ROC-AUC, confusion matrix

Run:  python train.py            (full run, a few minutes on CPU)
      python train.py --trials 4 (quicker hyperparameter search)
"""
import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from common import (ALL_FEATURES, CATEGORICAL_FEATURES, CLASS_NAMES, CLASS_RULES, FEATURE_INFO, FIG_DIR, INK,
                    INK_2, MODELS_DIR, NUMERIC_FEATURES, OUTPUTS_DIR, SEED, SERIES, SURFACE, TARGET,
                    apply_chart_style, grade_to_class, load_data)

N_SELECTED = 15      # candidate features kept by the filter ranking (stage 1)
PRUNE_THRESHOLD = 0.002   # stage 2: keep a candidate only if shuffling it lowers CV macro F1 by more than this
PRUNE_REPEATS = 10
PRUNE_CFG = {"hidden_units": (64, 32), "dropout": 0.2, "l2": 1e-3, "learning_rate": 1e-3,
             "batch_size": 32, "batch_norm": False}
TEST_SIZE = 0.20
MAX_EPOCHS = 200
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#86b6ef", "#2a78d6", "#0d366b"])


# --------------------------------------------------------------------------- preprocessing
def make_preprocessor(features):
    """Standardise numeric/ordinal features and one-hot encode nominal ones."""
    num = [f for f in features if f in NUMERIC_FEATURES]
    cat = [f for f in features if f in CATEGORICAL_FEATURES]
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), cat),
    ])


# --------------------------------------------------------------------------- feature selection
def select_features(X_train, y_train, k):
    """Rank raw features by mutual information and random-forest importance; keep the top k."""
    enc = X_train.copy()
    enc[CATEGORICAL_FEATURES] = OrdinalEncoder().fit_transform(enc[CATEGORICAL_FEATURES])
    discrete = [c != "absences" for c in enc.columns]

    mi = np.mean([mutual_info_classif(enc, y_train, discrete_features=discrete, random_state=s)
                  for s in range(5)], axis=0)
    rf = RandomForestClassifier(n_estimators=500, random_state=SEED, class_weight="balanced", n_jobs=-1)
    rf.fit(enc, y_train)

    rank = pd.DataFrame({"feature": enc.columns, "mutual_info": mi, "rf_importance": rf.feature_importances_})
    rank["mi_rank"] = rank["mutual_info"].rank(ascending=False)
    rank["rf_rank"] = rank["rf_importance"].rank(ascending=False)
    rank["mean_rank"] = (rank["mi_rank"] + rank["rf_rank"]) / 2
    rank = rank.sort_values(["mean_rank", "rf_rank"]).reset_index(drop=True)
    rank["selected"] = rank.index < k
    selected = [f for f in ALL_FEATURES if f in set(rank.loc[rank["selected"], "feature"])]
    return selected, rank


# --------------------------------------------------------------------------- DNN
def build_dnn(n_inputs, hidden_units, dropout, l2, learning_rate, batch_norm, **_):
    model = keras.Sequential(name="student_performance_dnn")
    model.add(keras.Input(shape=(n_inputs,)))
    for units in hidden_units:
        model.add(layers.Dense(units, kernel_regularizer=regularizers.l2(l2)))
        if batch_norm:
            model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(len(CLASS_NAMES), activation="softmax"))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def fit_dnn(params, X_tr, y_tr, X_val, y_val, seed=SEED):
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    weights = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    model = build_dnn(X_tr.shape[1], **params)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5),
    ]
    history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=MAX_EPOCHS,
                        batch_size=params["batch_size"], class_weight=dict(enumerate(weights)),
                        callbacks=callbacks, verbose=0)
    return model, history


def sample_configs(n, rng):
    """Random search space for the DNN."""
    space = {
        "hidden_units": [(32,), (64,), (64, 32), (128, 64), (64, 32, 16), (128, 64, 32)],
        "dropout": [0.1, 0.2, 0.3, 0.4],
        "l2": [0.0, 1e-4, 1e-3, 1e-2],
        "learning_rate": [3e-4, 1e-3, 3e-3],
        "batch_size": [16, 32],
        "batch_norm": [True, False],
    }
    configs, seen = [], set()
    while len(configs) < n:
        cfg = {k: v[rng.integers(len(v))] for k, v in space.items()}
        key = json.dumps(cfg, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            configs.append(cfg)
    return configs


def prune_features(X_train, y_train, features):
    """Stage 2 of feature selection: permutation importance of each candidate for a DNN.

    3-fold CV on the training data only: train a DNN on two folds, then shuffle one feature at a time in the
    third fold and measure the drop in macro F1. Features whose shuffling does not hurt are removed.
    """
    rng = np.random.default_rng(SEED)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    drops = {f: [] for f in features}
    for tr_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val, y_val = X_train.iloc[tr_idx], X_train.iloc[val_idx], y_train[val_idx]
        pre = make_preprocessor(features)
        model, _ = fit_dnn(PRUNE_CFG, pre.fit_transform(X_tr), y_train[tr_idx], pre.transform(X_val), y_val)
        score = lambda d: f1_score(y_val, model(pre.transform(d), training=False).numpy().argmax(1),
                                   average="macro")
        base = score(X_val)
        for f in features:
            for _ in range(PRUNE_REPEATS):
                shuffled = X_val.copy()
                shuffled[f] = rng.permutation(shuffled[f].values)
                drops[f].append(base - score(shuffled))
    importance = pd.Series({f: float(np.mean(v)) for f, v in drops.items()})
    kept = [f for f in features if importance[f] > PRUNE_THRESHOLD]
    return kept, importance


def tune_dnn(X_train, y_train, features, n_trials):
    """Random search with 3-fold stratified cross-validation; preprocessing is refit inside every fold."""
    rng = np.random.default_rng(SEED)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    rows = []
    for t, cfg in enumerate(sample_configs(n_trials, rng), 1):
        f1s, accs, epochs = [], [], []
        for tr_idx, val_idx in skf.split(X_train, y_train):
            pre = make_preprocessor(features)
            X_tr = pre.fit_transform(X_train.iloc[tr_idx])
            X_val = pre.transform(X_train.iloc[val_idx])
            model, hist = fit_dnn(cfg, X_tr, y_train[tr_idx], X_val, y_train[val_idx])
            pred = model.predict(X_val, verbose=0).argmax(1)
            f1s.append(f1_score(y_train[val_idx], pred, average="macro"))
            accs.append(accuracy_score(y_train[val_idx], pred))
            epochs.append(int(np.argmin(hist.history["val_loss"])) + 1)
        rows.append({**cfg, "hidden_units": "-".join(map(str, cfg["hidden_units"])),
                     "cv_macro_f1": np.mean(f1s), "cv_macro_f1_std": np.std(f1s),
                     "cv_accuracy": np.mean(accs), "mean_best_epoch": np.mean(epochs), "_cfg": cfg})
        print(f"  trial {t:>2}/{n_trials}  layers={rows[-1]['hidden_units']:<10} dropout={cfg['dropout']:<4}"
              f" l2={cfg['l2']:<6} lr={cfg['learning_rate']:<6} batch={cfg['batch_size']:<3}"
              f" bn={str(cfg['batch_norm']):<5} -> CV macro-F1 {np.mean(f1s):.4f}  acc {np.mean(accs):.4f}")
    results = pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False).reset_index(drop=True)
    best = results.loc[0, "_cfg"]
    return best, results.drop(columns="_cfg")


# --------------------------------------------------------------------------- evaluation
def evaluate(name, y_true, pred, proba):
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, pred),
        "precision_macro": precision_score(y_true, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, pred, average="macro"),
        "f1_weighted": f1_score(y_true, pred, average="weighted"),
        "roc_auc_ovr": roc_auc_score(y_true, proba, multi_class="ovr", average="macro"),
    }


# --------------------------------------------------------------------------- figures
def save(fig, name):
    fig.savefig(FIG_DIR / name)
    plt.close(fig)
    print(f"  saved outputs/figures/{name}")


def plot_feature_ranking(rank):
    r = rank.iloc[::-1]
    colors = [SERIES[0] if s else SERIES[3] if c else "#c3c2b7" for s, c in zip(r["selected"], r["candidate"])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 8), sharey=True)
    panels = [("mutual_info", "1a. Mutual information"), ("rf_importance", "1b. Random-forest importance"),
              ("perm_importance", "2. DNN permutation importance\n(candidates only, CV on training data)")]
    for ax, (col, title) in zip(axes, panels):
        ax.barh(r["feature"], r[col].fillna(0), color=colors, height=0.7)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[2].axvline(PRUNE_THRESHOLD, color=INK_2, linewidth=0.8, linestyle="--")
    axes[2].axvline(0, color=INK_2, linewidth=0.8)
    n_cand = int(rank["candidate"].sum())
    axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (SERIES[0], SERIES[3], "#c3c2b7")],
                   labels=[f"Selected ({int(rank['selected'].sum())})",
                           f"Candidate, dropped in stage 2 ({n_cand - int(rank['selected'].sum())})",
                           f"Dropped in stage 1 ({len(rank) - n_cand})"], loc="lower right")
    fig.suptitle("Feature selection: ranking of all 32 input attributes", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "07_feature_selection.png")


def plot_history(history):
    h = history.history
    best = int(np.argmin(h["val_loss"]))
    x = np.arange(1, len(h["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in zip(axes, ["loss", "accuracy"], ["Loss (cross-entropy)", "Accuracy"]):
        ax.plot(x, h[key], color=SERIES[0], label="Training")
        ax.plot(x, h["val_" + key], color=SERIES[1], label="Validation")
        ax.axvline(best + 1, color=INK_2, linewidth=0.8, linestyle="--")
        ax.set(title=title, xlabel="Epoch")
    axes[0].annotate(f"best epoch = {best + 1}\n(weights restored)", xy=(best + 1, max(h["loss"])),
                     xytext=(6, -4), textcoords="offset points", va="top", fontsize=8, color=INK_2)
    axes[1].legend(loc="lower right")
    fig.suptitle("DNN training curves", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "08_dnn_training_curves.png")


def plot_confusions(y_true, preds):
    fig, axes = plt.subplots(1, len(preds), figsize=(4.3 * len(preds), 4.3))
    for ax, (name, pred) in zip(np.atleast_1d(axes), preds.items()):
        cm = confusion_matrix(y_true, pred)
        ax.imshow(cm, cmap=SEQUENTIAL, vmin=0, vmax=cm.max())
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}\n{cm[i, j] / cm[i].sum():.0%}", ha="center", va="center",
                        fontsize=9, color="white" if cm[i, j] > cm.max() * 0.5 else INK)
        ax.set_xticks(range(3), CLASS_NAMES)
        ax.set_yticks(range(3), CLASS_NAMES)
        ax.set_xticks(np.arange(-.5, 3), minor=True)
        ax.set_yticks(np.arange(-.5, 3), minor=True)
        ax.grid(False)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set(title=f"{name}\naccuracy {accuracy_score(y_true, pred):.1%}", xlabel="Predicted")
    np.atleast_1d(axes)[0].set_ylabel("Actual")
    fig.suptitle("Confusion matrices on the held-out test set (count, % of actual class)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "09_confusion_matrices.png")


def plot_comparison(table):
    t = table.sort_values("f1_macro")
    metrics = [("accuracy", "Accuracy"), ("f1_macro", "Macro F1"), ("roc_auc_ovr", "ROC-AUC (one-vs-rest)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, (col, title) in zip(axes, metrics):
        colors = [SERIES[0] if "DNN" in m else "#a9a8a1" for m in t["model"]]
        bars = ax.barh(t["model"], t[col], color=colors, height=0.6)
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in t[col]], padding=4, color=INK, fontsize=9)
        ax.set(title=title, xlim=(0, 1.12))
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="y", visible=False)
    fig.suptitle("Deep Neural Network vs traditional ML on the held-out test set", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "10_model_comparison.png")


# --------------------------------------------------------------------------- main
def main(n_trials):
    t0 = time.time()
    for d in (MODELS_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    apply_chart_style()
    keras.utils.set_random_seed(SEED)

    # 1. Data + preprocessing -------------------------------------------------
    df = load_data()
    print(f"[1/6] Loaded {len(df)} records, {df.shape[1] - 1} input attributes. "
          f"Missing values: {int(df.isna().sum().sum())}, duplicates: {int(df.duplicated().sum())}")
    X = df[ALL_FEATURES]
    y = df[TARGET].map(grade_to_class).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    print(f"      Train {len(X_train)} / test {len(X_test)} (stratified). "
          f"Class counts (train): {dict(zip(CLASS_NAMES, np.bincount(y_train).tolist()))}")

    # 2. Feature selection ----------------------------------------------------
    candidates, rank = select_features(X_train, y_train, N_SELECTED)
    print(f"[2/6] Stage 1 (mutual information + RF importance): {len(candidates)} candidates: {candidates}")
    features, perm = prune_features(X_train[candidates], y_train, candidates)
    rank["perm_importance"] = rank["feature"].map(perm)
    rank["candidate"] = rank["feature"].isin(candidates)
    rank["selected"] = rank["feature"].isin(features)
    rank.round(4).to_csv(OUTPUTS_DIR / "feature_ranking.csv", index=False)
    print("      Stage 2 (DNN permutation importance, 3-fold CV on training data):")
    for f, v in perm.sort_values(ascending=False).items():
        print(f"        {f:<10} {v:+.4f}  {'keep' if f in features else 'drop'}")
    print(f"      Selected {len(features)} of {len(ALL_FEATURES)} features: {features}")
    X_train, X_test = X_train[features].reset_index(drop=True), X_test[features].reset_index(drop=True)

    # 3. Traditional ML baselines --------------------------------------------
    print("[3/6] Traditional ML models (grid search, 5-fold CV, scoring = macro F1)")
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    grids = {
        "Logistic Regression": (LogisticRegression(max_iter=5000, random_state=SEED),
                                {"clf__C": [0.01, 0.1, 1, 10], "clf__class_weight": [None, "balanced"]}),
        "Random Forest": (RandomForestClassifier(random_state=SEED, n_jobs=-1),
                          {"clf__n_estimators": [200, 500], "clf__max_depth": [None, 6, 10],
                           "clf__min_samples_leaf": [1, 3, 5], "clf__class_weight": [None, "balanced"]}),
        "SVM (RBF)": (SVC(probability=True, random_state=SEED),
                      {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1],
                       "clf__class_weight": [None, "balanced"]}),
    }
    results, preds, cv_scores, best_params = [], {}, {}, {}
    for name, (clf, grid) in grids.items():
        gs = GridSearchCV(Pipeline([("pre", make_preprocessor(features)), ("clf", clf)]), grid,
                          scoring="f1_macro", cv=cv5, n_jobs=-1)
        gs.fit(X_train, y_train)
        preds[name] = gs.predict(X_test)
        results.append(evaluate(name, y_test, preds[name], gs.predict_proba(X_test)))
        cv_scores[name] = gs.best_score_
        best_params[name] = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
        print(f"  {name:<20} CV macro-F1 {gs.best_score_:.4f}  best params: {best_params[name]}")
        joblib.dump(gs.best_estimator_, MODELS_DIR / f"{name.split()[0].lower()}_pipeline.joblib")

    # 4. DNN hyperparameter tuning -------------------------------------------
    print(f"[4/6] DNN hyperparameter tuning (random search, {n_trials} trials x 3-fold CV)")
    best_cfg, tuning = tune_dnn(X_train, y_train, features, n_trials)
    tuning.round(4).to_csv(OUTPUTS_DIR / "dnn_tuning_results.csv", index=False)
    cv_scores["Deep Neural Network (DNN)"] = float(tuning.loc[0, "cv_macro_f1"])
    print(f"      Best configuration: {best_cfg}")

    # 5. Final DNN ------------------------------------------------------------
    print("[5/6] Training the final DNN with the best hyperparameters")
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train,
                                                  random_state=SEED)
    preprocessor = make_preprocessor(features)
    X_fit_t, X_val_t = preprocessor.fit_transform(X_fit), preprocessor.transform(X_val)
    model, history = fit_dnn(best_cfg, X_fit_t, y_fit, X_val_t, y_val)
    model.summary()

    # 6. Evaluation -----------------------------------------------------------
    proba = model.predict(preprocessor.transform(X_test), verbose=0)
    preds["Deep Neural Network (DNN)"] = proba.argmax(1)
    results.append(evaluate("Deep Neural Network (DNN)", y_test, proba.argmax(1), proba))
    table = pd.DataFrame(results)
    table.insert(1, "cv_f1_macro", table["model"].map(cv_scores))
    table = table.sort_values("f1_macro", ascending=False).reset_index(drop=True)
    table.round(4).to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)

    print("[6/6] Held-out test set results")
    print(table.round(4).to_string(index=False))
    report = classification_report(y_test, proba.argmax(1), target_names=CLASS_NAMES, digits=4)
    print("\nDNN classification report (test set)\n" + report)
    (OUTPUTS_DIR / "dnn_classification_report.txt").write_text(report, encoding="utf-8")

    print("Figures:")
    plot_feature_ranking(rank)
    plot_history(history)
    best_ml = max((r for r in results if "DNN" not in r["model"]), key=lambda r: r["f1_macro"])["model"]
    plot_confusions(y_test, {"Deep Neural Network (DNN)": preds["Deep Neural Network (DNN)"],
                             **{m: p for m, p in preds.items() if "DNN" not in m}})
    plot_comparison(table)

    # Artifacts for the Streamlit app ----------------------------------------
    model.save(MODELS_DIR / "dnn_model.keras")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    defaults = {f: (X[f].mode()[0] if f in CATEGORICAL_FEATURES or "options" in FEATURE_INFO[f]
                    else float(X[f].median())) for f in features}
    metadata = {
        "dataset": "UCI Student Performance - Portuguese course (student-por.csv)",
        "n_records": len(df), "n_train": len(X_train), "n_test": len(X_test),
        "class_names": CLASS_NAMES, "class_rules": CLASS_RULES,
        "selected_features": features,
        "defaults": {k: (v.item() if hasattr(v, "item") else v) for k, v in defaults.items()},
        "dnn_hyperparameters": {**best_cfg, "hidden_units": list(best_cfg["hidden_units"])},
        "dnn_epochs_trained": len(history.history["loss"]),
        "ml_best_params": best_params, "best_traditional_model": best_ml,
        "test_metrics": {r["model"]: {k: round(float(v), 6) for k, v in r.items() if k != "model"} for r in results},
        "cv_macro_f1": {k: round(float(v), 4) for k, v in cv_scores.items()},
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved models/dnn_model.keras, models/preprocessor.joblib, models/metadata.json"
          f"  (total time {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=16, help="number of DNN random-search trials")
    main(parser.parse_args().trials)
