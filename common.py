"""Shared configuration and helpers for the Student Academic Performance project."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "student-por.csv"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
FIG_DIR = OUTPUTS_DIR / "figures"

SEED = 42
TARGET = "G3"

# Final grade (0-20) -> performance category
CLASS_NAMES = ["Low", "Medium", "High"]
CLASS_RULES = {"Low": "G3 0-9 (fail)", "Medium": "G3 10-13 (pass)", "High": "G3 14-20 (good / excellent)"}

# Chart styling (one colour per class, fixed order)
CLASS_COLORS = {"Low": "#2a78d6", "Medium": "#eb6834", "High": "#1baf7a"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

CATEGORICAL_FEATURES = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian",
    "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic",
]
NUMERIC_FEATURES = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime",
    "goout", "Dalc", "Walc", "health", "absences", "G1", "G2",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

_YES_NO = {"yes": "Yes", "no": "No"}
_JOBS = {"teacher": "Teacher", "health": "Health care", "services": "Civil services",
         "at_home": "At home", "other": "Other"}
_EDU = {0: "None", 1: "Primary (4th grade)", 2: "5th to 9th grade", 3: "Secondary", 4: "Higher education"}
_SCALE5 = {1: "1 - Very low", 2: "2 - Low", 3: "3 - Medium", 4: "4 - High", 5: "5 - Very high"}

# Describes every input attribute: label, group and allowed values.
# "options" maps the raw dataset value -> human readable label (used by the Streamlit form).
FEATURE_INFO = {
    "school": {"label": "School", "group": "Demographic",
               "options": {"GP": "Gabriel Pereira", "MS": "Mousinho da Silveira"}},
    "sex": {"label": "Sex", "group": "Demographic", "options": {"F": "Female", "M": "Male"}},
    "age": {"label": "Age", "group": "Demographic", "min": 15, "max": 22},
    "address": {"label": "Home address type", "group": "Demographic", "options": {"U": "Urban", "R": "Rural"}},
    "famsize": {"label": "Family size", "group": "Demographic",
                "options": {"LE3": "3 or fewer", "GT3": "More than 3"}},
    "Pstatus": {"label": "Parents' cohabitation status", "group": "Demographic",
                "options": {"T": "Living together", "A": "Apart"}},
    "Medu": {"label": "Mother's education", "group": "Demographic", "options": _EDU},
    "Fedu": {"label": "Father's education", "group": "Demographic", "options": _EDU},
    "Mjob": {"label": "Mother's job", "group": "Demographic", "options": _JOBS},
    "Fjob": {"label": "Father's job", "group": "Demographic", "options": _JOBS},
    "guardian": {"label": "Guardian", "group": "Demographic",
                 "options": {"mother": "Mother", "father": "Father", "other": "Other"}},
    "reason": {"label": "Reason for choosing this school", "group": "Academic",
               "options": {"home": "Close to home", "reputation": "School reputation",
                           "course": "Course preference", "other": "Other"}},
    "traveltime": {"label": "Home-to-school travel time", "group": "Academic",
                   "options": {1: "Under 15 min", 2: "15 to 30 min", 3: "30 min to 1 hour", 4: "Over 1 hour"}},
    "studytime": {"label": "Weekly study time", "group": "Academic",
                  "options": {1: "Under 2 hours", 2: "2 to 5 hours", 3: "5 to 10 hours", 4: "Over 10 hours"}},
    "failures": {"label": "Past class failures", "group": "Academic",
                 "options": {0: "0", 1: "1", 2: "2", 3: "3 or more"}},
    "schoolsup": {"label": "Extra educational support (school)", "group": "Academic", "options": _YES_NO},
    "famsup": {"label": "Family educational support", "group": "Academic", "options": _YES_NO},
    "paid": {"label": "Extra paid classes", "group": "Academic", "options": _YES_NO},
    "nursery": {"label": "Attended nursery school", "group": "Academic", "options": _YES_NO},
    "higher": {"label": "Wants higher education", "group": "Academic", "options": _YES_NO},
    "absences": {"label": "Number of school absences", "group": "Academic", "min": 0, "max": 93},
    "G1": {"label": "First period grade (G1, 0-20)", "group": "Academic", "min": 0, "max": 20},
    "G2": {"label": "Second period grade (G2, 0-20)", "group": "Academic", "min": 0, "max": 20},
    "activities": {"label": "Extra-curricular activities", "group": "Social & behavioural", "options": _YES_NO},
    "internet": {"label": "Internet access at home", "group": "Social & behavioural", "options": _YES_NO},
    "romantic": {"label": "In a romantic relationship", "group": "Social & behavioural", "options": _YES_NO},
    "famrel": {"label": "Quality of family relationships", "group": "Social & behavioural",
               "options": {1: "1 - Very bad", 2: "2 - Bad", 3: "3 - Average", 4: "4 - Good", 5: "5 - Excellent"}},
    "freetime": {"label": "Free time after school", "group": "Social & behavioural", "options": _SCALE5},
    "goout": {"label": "Going out with friends", "group": "Social & behavioural", "options": _SCALE5},
    "Dalc": {"label": "Workday alcohol consumption", "group": "Social & behavioural", "options": _SCALE5},
    "Walc": {"label": "Weekend alcohol consumption", "group": "Social & behavioural", "options": _SCALE5},
    "health": {"label": "Current health status", "group": "Social & behavioural",
               "options": {1: "1 - Very bad", 2: "2 - Bad", 3: "3 - Average", 4: "4 - Good", 5: "5 - Very good"}},
}


def load_data() -> pd.DataFrame:
    """Load the UCI Student Performance (Portuguese course) dataset."""
    return pd.read_csv(DATA_PATH, sep=";")


def grade_to_class(g3) -> int:
    """Map a final grade (0-20) to a class index: 0 = Low, 1 = Medium, 2 = High."""
    if g3 < 10:
        return 0
    if g3 < 14:
        return 1
    return 2


def apply_chart_style():
    """Consistent, quiet matplotlib styling for every figure in the project."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 10, "text.color": INK, "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.major.size": 0, "ytick.major.size": 0, "lines.linewidth": 2,
        "legend.frameon": False, "axes.prop_cycle": mpl.cycler(color=SERIES),
    })
