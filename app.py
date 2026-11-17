"""Streamlit web app: Student Academic Performance Prediction.

Run:  streamlit run app.py     (after `python train.py` has produced the models/ folder)
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import joblib
import pandas as pd
import streamlit as st

from common import CLASS_NAMES, CLASS_RULES, FEATURE_INFO, FIG_DIR, MODELS_DIR, OUTPUTS_DIR

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

# Performance category -> employability readiness band shown to the user
EMPLOYABILITY = {
    "Low": ("Needs significant support", "error",
            "The student is predicted to fail the course. Recommend remedial classes, mentoring and close "
            "monitoring of attendance before placement activities."),
    "Medium": ("Employable with upskilling", "warning",
               "The student is predicted to pass with an average grade. Targeted skill-building and extra "
               "study time would strengthen the employability profile."),
    "High": ("Highly employable", "success",
             "The student is predicted to finish with a good or excellent grade and shows a strong academic "
             "profile for employability."),
}


@st.cache_resource(show_spinner="Loading the trained model...")
def load_artifacts():
    from tensorflow import keras

    model = keras.models.load_model(MODELS_DIR / "dnn_model.keras")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    metadata = json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))
    return model, preprocessor, metadata


def input_widget(feature, default):
    """Render the right widget for one feature and return the raw dataset value."""
    info = FEATURE_INFO[feature]
    if "options" in info:
        keys = list(info["options"].keys())
        index = keys.index(default) if default in keys else 0
        return st.selectbox(info["label"], keys, index=index, format_func=lambda k: info["options"][k], key=feature)
    return st.number_input(info["label"], min_value=info["min"], max_value=info["max"],
                           value=int(default), step=1, key=feature)


def prediction_tab(model, preprocessor, meta):
    features = meta["selected_features"]
    st.write("Enter the student's details and press **Predict** to estimate the final academic performance "
             "category. Only the attributes kept by feature selection are requested.")

    with st.form("student_form"):
        values = {}
        groups = ["Demographic", "Academic", "Social & behavioural"]
        columns = st.columns(len(groups))
        for col, group in zip(columns, groups):
            with col:
                st.subheader(group)
                group_features = [f for f in features if FEATURE_INFO[f]["group"] == group]
                if not group_features:
                    st.caption("No attributes from this group were selected.")
                for f in group_features:
                    values[f] = input_widget(f, meta["defaults"][f])
        submitted = st.form_submit_button("Predict performance", type="primary", width="stretch")

    if not submitted:
        return

    row = pd.DataFrame([values])[features]
    proba = model.predict(preprocessor.transform(row), verbose=0)[0]
    label = CLASS_NAMES[int(proba.argmax())]
    band, level, advice = EMPLOYABILITY[label]

    st.divider()
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Prediction")
        st.metric("Predicted performance category", label, help=CLASS_RULES[label])
        st.caption(f"Expected final grade range: {CLASS_RULES[label]}")
        getattr(st, level)(f"**Employability grade: {band}**\n\n{advice}")
    with right:
        st.subheader("Class probabilities")
        for name, p in zip(CLASS_NAMES, proba):
            st.progress(float(p), text=f"{name} ({CLASS_RULES[name]}): {p:.1%}")
        st.caption(f"Model confidence: {proba.max():.1%}")
    st.caption("This is a statistical estimate from a model trained on 649 Portuguese secondary-school students. "
               "Use it to guide support, not as the sole basis for decisions about a student.")


def performance_tab(meta):
    st.subheader("Model comparison on the held-out test set")
    comparison = OUTPUTS_DIR / "model_comparison.csv"
    if comparison.exists():
        table = pd.read_csv(comparison).rename(columns={
            "model": "Model", "cv_f1_macro": "CV macro F1", "accuracy": "Accuracy",
            "precision_macro": "Precision (macro)", "recall_macro": "Recall (macro)", "f1_macro": "F1 (macro)",
            "f1_weighted": "F1 (weighted)", "roc_auc_ovr": "ROC-AUC (OvR)"})
        st.dataframe(table.style.format({c: "{:.3f}" for c in table.columns if c != "Model"}),
                     hide_index=True, width="stretch")
    st.caption(f"Trained on {meta['n_train']} students, tested on {meta['n_test']} unseen students.")

    hp = meta["dnn_hyperparameters"]
    st.subheader("Tuned DNN architecture")
    st.markdown(
        f"- **Hidden layers:** {' → '.join(str(u) for u in hp['hidden_units'])} (ReLU)\n"
        f"- **Dropout:** {hp['dropout']} · **L2:** {hp['l2']} · **Batch normalisation:** {hp['batch_norm']}\n"
        f"- **Optimiser:** Adam, learning rate {hp['learning_rate']} · **Batch size:** {hp['batch_size']}\n"
        f"- **Output:** 3-unit softmax (Low / Medium / High)")

    for file, caption in [("10_model_comparison.png", "DNN vs traditional ML"),
                          ("09_confusion_matrices.png", "Confusion matrices"),
                          ("08_dnn_training_curves.png", "DNN training curves"),
                          ("07_feature_selection.png", "Feature selection")]:
        if (FIG_DIR / file).exists():
            st.image(str(FIG_DIR / file), caption=caption, width="stretch")


def main():
    st.title("🎓 Student Academic Performance Prediction")
    st.caption("Deep Neural Network trained on the UCI Student Performance dataset (Portuguese course)")

    if not (MODELS_DIR / "dnn_model.keras").exists():
        st.error("No trained model found. Run `python train.py` first, then reload this page.")
        st.stop()
    model, preprocessor, meta = load_artifacts()

    with st.sidebar:
        st.header("About the model")
        dnn = meta["test_metrics"]["Deep Neural Network (DNN)"]
        st.metric("DNN test accuracy", f"{dnn['accuracy']:.1%}")
        st.metric("DNN macro F1", f"{dnn['f1_macro']:.3f}")
        st.markdown("**Performance categories**")
        for name in CLASS_NAMES:
            st.markdown(f"- **{name}** — {CLASS_RULES[name]}")
        st.markdown(f"**Inputs used:** {len(meta['selected_features'])} selected features")

    tab_predict, tab_perf = st.tabs(["Predict", "Model performance"])
    with tab_predict:
        prediction_tab(model, preprocessor, meta)
    with tab_perf:
        performance_tab(meta)


if __name__ == "__main__":
    main()
