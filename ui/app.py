
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px


st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")
st.title("❤️ Heart Disease Prediction")
st.caption("This app uses a trained pipeline (preprocess + feature selection + RandomForest) to predict heart disease risk from raw clinical inputs.")




from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

def simple_name_from_pre_name(name: str) -> str:
    """
    Convert ColumnTransformer output names like:
      'num__thalach' -> 'thalach'
      'cat__cp_4.0'  -> 'cp_4.0'
    """
    parts = name.split("__")
    return parts[-1]

class KeepBySimpleName(BaseEstimator, TransformerMixin):
    """
    Custom selector used in the saved pipeline.
    Expects that set_pre_names() is called by the training code to provide the
    ColumnTransformer output names. On transform, it keeps only the columns whose
    *simple* names match the selected set from step 2.3.
    """
    def __init__(self, keep_simple_names):
        self.keep_set = set(keep_simple_names)
        self.pre_names_ = None
        self.indices_ = None
        self.simple_names_ = None

    def set_pre_names(self, names):
        self.pre_names_ = list(names)
        return self

    def fit(self, X, y=None):
        if self.pre_names_ is None:
            raise RuntimeError("set_pre_names(names) must be called before fit().")
        self.simple_names_ = [simple_name_from_pre_name(n) for n in self.pre_names_]
        name_to_idx = {n: i for i, n in enumerate(self.simple_names_)}
        missing = [n for n in self.keep_set if n not in name_to_idx]
        if missing:
            # same behavior as training-time code
            raise ValueError(f"Missing in preprocessed output: {missing[:10]} ...")
        self.indices_ = np.array([name_to_idx[n] for n in self.keep_set], dtype=int)
        return self

    def transform(self, X):
        return X[:, self.indices_]
# --- end custom class block ---


# ---------- Paths & loaders ----------
ROOT = Path(__file__).resolve().parents[1]  # project root
MODEL_PATH = ROOT / "models" / "final_model.pkl"
DATA_PATH  = ROOT / "data"  / "heart_disease.csv"

@st.cache_resource
def load_model():
    import joblib
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    if "target" not in df.columns and "num" in df.columns:
        df["target"] = (df["num"] > 0).astype(int)
        df = df.drop(columns=["num"])
    df = df.replace("?", np.nan)
    for c in df.columns:
        if c != "target":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

pipe = load_model()
df_raw = load_data()


# ---------- Sidebar: inputs ----------
st.sidebar.header("Patient Inputs")

# Categorical dictionaries (UCI Heart)
sex_map    = {"Female (0)": 0, "Male (1)": 1}
cp_map     = {"Typical angina (1)": 1, "Atypical angina (2)": 2, "Non-anginal pain (3)": 3, "Asymptomatic (4)": 4}
fbs_map    = {"> 120 mg/dl (1)": 1, "≤ 120 mg/dl (0)": 0}
restecg_map= {"Normal (0)": 0, "ST-T abnormality (1)": 1, "LV hypertrophy (2)": 2}
exang_map  = {"Yes (1)": 1, "No (0)": 0}
slope_map  = {"Upsloping (1)": 1, "Flat (2)": 2, "Downsloping (3)": 3}
# In datasets used: ca ∈ {0,1,2,3,4}, thal ∈ {3,6,7}
ca_vals    = [0,1,2,3,4]
thal_map   = {"Normal (3)": 3, "Fixed defect (6)": 6, "Reversible defect (7)": 7}

# Numeric sliders based on typical ranges
age       = st.sidebar.slider("Age (years)", 18, 100, 54)
trestbps  = st.sidebar.slider("Resting blood pressure (mm Hg)", 80, 200, 130)
chol      = st.sidebar.slider("Serum cholesterol (mg/dl)", 100, 600, 246)
thalach   = st.sidebar.slider("Max heart rate achieved", 60, 220, 150)
oldpeak   = st.sidebar.slider("ST depression induced by exercise", 0.0, 6.5, 1.0, step=0.1)

sex       = sex_map[ st.sidebar.selectbox("Sex", tuple(sex_map.keys())) ]
cp        = cp_map[  st.sidebar.selectbox("Chest pain type", tuple(cp_map.keys())) ]
fbs       = fbs_map[ st.sidebar.selectbox("Fasting blood sugar", tuple(fbs_map.keys())) ]
restecg   = restecg_map[ st.sidebar.selectbox("Resting ECG results", tuple(restecg_map.keys())) ]
exang     = exang_map[ st.sidebar.selectbox("Exercise-induced angina", tuple(exang_map.keys())) ]
slope     = slope_map[ st.sidebar.selectbox("Slope of ST segment", tuple(slope_map.keys())) ]
ca        = st.sidebar.selectbox("Number of major vessels (0–4) colored by flourosopy", ca_vals)
thal      = thal_map[ st.sidebar.selectbox("Thalassemia", tuple(thal_map.keys())) ]

# Build a single-row RAW DataFrame 
row = pd.DataFrame([{
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol, "fbs": fbs,
    "restecg": restecg, "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
    "slope": slope, "ca": ca, "thal": thal
}])

# ---------- Prediction ----------
st.subheader("Prediction")
with st.spinner("Scoring…"):
    prob = float(pipe.predict_proba(row)[0,1])
    pred = int(pipe.predict(row)[0])

col1, col2 = st.columns(2)
with col1:
    st.metric("Predicted probability of heart disease", f"{prob:.3f}")
with col2:
    st.metric("Predicted class (0 = No disease, 1 = Disease)", str(pred))

st.caption("This probability is computed by the final trained model. Inputs are preprocessed (imputation, scaling, one-hot encoding) inside the pipeline.")

# ---------- Data exploration ----------
st.markdown("---")
st.header("Explore Dataset")

tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Relationships", "Chest Pain Types", "Correlation"])

with tab1:
    c1, c2, c3 = st.columns(3)
    fig_age = px.histogram(df_raw, x="age", color="target", barmode="overlay", nbins=20, title="Age distribution by target")
    c1.plotly_chart(fig_age, use_container_width=True)
    fig_chol = px.histogram(df_raw, x="chol", color="target", barmode="overlay", nbins=30, title="Cholesterol by target")
    c2.plotly_chart(fig_chol, use_container_width=True)
    fig_thalach = px.histogram(df_raw, x="thalach", color="target", barmode="overlay", nbins=25, title="Max heart rate by target")
    c3.plotly_chart(fig_thalach, use_container_width=True)

with tab2:
    fig_scatter = px.scatter(df_raw, x="age", y="chol", color="target", title="Age vs. Cholesterol")
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    cp_counts = df_raw.groupby(["cp","target"]).size().reset_index(name="count")
    fig_cp = px.bar(cp_counts, x="cp", y="count", color="target", barmode="group", title="Chest pain type counts by target")
    st.plotly_chart(fig_cp, use_container_width=True)

with tab4:
    num_cols = ["age","trestbps","chol","thalach","oldpeak"]
    corr = df_raw[num_cols + ["target"]].corr(numeric_only=True)
    fig_corr = px.imshow(corr, text_auto=True, title="Correlation (numeric features)")
    st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("---")
st.info("Tip: use the left sidebar to change inputs and see the prediction update in real time.")

