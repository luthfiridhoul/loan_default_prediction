
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

st.set_page_config(
    page_title="Loan Default Prediction Studio",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------- Session helpers --------
def _store_model_to_session(clf_obj, selected_features, target, meta=None):
    st.session_state["clf"] = clf_obj
    st.session_state["selected_features"] = list(selected_features)
    st.session_state["target"] = target
    st.session_state["trained"] = True
    if meta is not None:
        st.session_state["meta"] = meta

def _get_model_from_session():
    return st.session_state.get("clf", None)

# ----------------- THEME / STYLES -----------------
st.markdown("""
<style>
:root {
  --bg: #0e1117;
  --card: #111827;
  --text: #e5e7eb;
  --muted: #94a3b8;
  --accent1: #22d3ee;   /* cyan */
  --accent2: #a78bfa;   /* violet */
  --accent3: #34d399;   /* green */
  --danger: #f43f5e;    /* rose */
  --warning: #f59e0b;   /* amber */
}
html, body, [class*="css"]  {
  background-color: var(--bg);
  color: var(--text);
}
.block-container { padding-top: 1.2rem; }
div.stMetric { background: linear-gradient(135deg, #111827, #0b1220); border-radius: 16px; padding: 12px; border: 1px solid #1f2937;}
.stButton>button {
  border-radius: 10px;
  border: 1px solid #1f2937;
  background: linear-gradient(135deg, var(--accent2), var(--accent1));
  color: #0b0f16;
  font-weight: 700;
}
section[data-testid="stSidebar"] { background: #0b1220; }
</style>
""", unsafe_allow_html=True)

# ----------------- HEADER -----------------
lcol, rcol = st.columns([0.7, 0.3])
with lcol:
    st.title("💸 Loan Default Prediction Studio")
    st.caption("Dashboard interaktif untuk memprediksi pelanggan yang default.")
with rcol:
    st.write("")

# ----------------- SIDEBAR -----------------
st.sidebar.markdown("### 🔍 Data & Opsi")
uploaded = st.sidebar.file_uploader("Unggah CSV (contoh: Loan_default.csv)", type=["csv"])

# Decision threshold
decision_threshold = st.sidebar.slider("⚖️ Decision threshold (probabilitas default)",
                                       min_value=0.10, max_value=0.90, value=0.50, step=0.01)

# ----------------- LOAD DATA -----------------
def example_dataset():
    rng = np.random.default_rng(7)
    n = 800
    df = pd.DataFrame({
        "gender": rng.choice(["Female", "Male"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": rng.choice(["Yes", "No"], n, p=[0.45, 0.55]),
        "Dependents": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "tenure": rng.integers(0, 72, n),
        "PhoneService": rng.choice(["Yes", "No"], n, p=[0.9, 0.1]),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.4, 0.5, 0.1]),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.6, 0.25, 0.15]),
        "PaperlessBilling": rng.choice(["Yes", "No"], n, p=[0.7, 0.3]),
        "MonthlyCharges": rng.normal(75, 25, n).clip(15, 150).round(2),
    })
    df["TotalCharges"] = (df["MonthlyCharges"] * (df["tenure"].replace(0, 1))).round(2)
    p = 0.6*(df["Contract"].eq("Month-to-month")).astype(int) + 0.25*(df["tenure"] < 12).astype(int) + rng.random(n)*0.2
    df["Default"] = (p > 0.6).astype(int)
    return df

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    if os.path.exists("Loan_default.csv"):
        try:
            df = pd.read_csv("Loan_default.csv")
            st.info("Memuat **Loan_default.csv** dari folder lokal.")
        except Exception:
            st.info("Gagal membaca Loan_default.csv. Menggunakan dataset contoh.")
            df = example_dataset()
    else:
        st.info("Tidak ada file diunggah. Menggunakan dataset **contoh** untuk eksplorasi.")
        df = example_dataset()

with st.expander("👀 Lihat Sample Data", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

# ----------------- CONFIG -----------------
all_cols = df.columns.tolist()
default_target = "Default" if "Default" in all_cols else all_cols[-1]
target = st.sidebar.selectbox("🎯 Kolom Target (Default):", options=all_cols, index=all_cols.index(default_target))

feature_cols = [c for c in all_cols if c != target]
selected_features = st.sidebar.multiselect("🧩 Pilih Fitur (kosongkan = semua)", options=feature_cols, default=feature_cols)
if len(selected_features) == 0:
    selected_features = feature_cols

X = df[selected_features].copy()
y = df[target].copy()

# detect types
raw_cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
# skip high-cardinality categorical columns for EDA
cat_cols = [c for c in raw_cat_cols if df[c].nunique() <= 20]
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

st.markdown("### 🧪 Pra-pemrosesan & Splitting")
c1, c2, c3 = st.columns(3)
test_size = c1.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = c2.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)
scale_numeric = c3.checkbox("Standardize numeric features", value=True)

# preprocessors
num_steps = [("imputer", SimpleImputer(strategy="median"))]
if scale_numeric:
    num_steps.append(("scaler", StandardScaler()))
numeric_transformer = Pipeline(steps=num_steps)

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, raw_cat_cols)  # use raw here for modeling
    ]
)

# ----------------- MODEL -----------------
st.markdown("### 🤖 Pilih & Latih Model")
model_name = st.selectbox("Algoritma", ["Logistic Regression", "Random Forest", "XGBoost"])
base_model = None
auto_scale_pos_weight = False

if model_name == "Logistic Regression":
    C_val = st.slider("C (inverse regularization)", 0.01, 5.0, 1.0)
    base_model = LogisticRegression(max_iter=2000, C=C_val)
elif model_name == "Random Forest":
    n_estimators = st.slider("n_estimators", 50, 500, 200, 50)
    max_depth = st.slider("max_depth", 2, 20, 8)
    base_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=random_state)
elif model_name == "XGBoost":
    try:
        from xgboost import XGBClassifier
        n_estimators = st.slider("n_estimators", 50, 1000, 300, 50)
        max_depth = st.slider("max_depth", 2, 12, 6)
        learning_rate = st.slider("learning_rate", 0.01, 0.5, 0.1)
        subsample = st.slider("subsample", 0.5, 1.0, 0.8)
        colsample_bytree = st.slider("colsample_bytree", 0.5, 1.0, 0.8)
        auto_scale_pos_weight = st.checkbox("Auto scale_pos_weight (imbalance handling)", value=True)
        base_model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state
        )
    except Exception as e:
        st.error("XGBoost belum terpasang. Tambahkan 'xgboost' ke requirements.txt lalu install. Detail: {}".format(e))
        base_model = None

clf = Pipeline(steps=[("preprocess", preprocessor), ("model", base_model)]) if base_model is not None else None

# ----------------- TRAIN -----------------
train_btn = st.button("🚀 Train Model")
trained = st.session_state.get('trained', False)
if train_btn:
    if clf is None:
        st.error("Model belum siap. Pastikan dependensi terpasang dengan benar.")
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state,
                stratify=y if len(np.unique(y))==2 else None
            )
            # handle imbalance for XGBoost if selected
            if model_name == "XGBoost" and auto_scale_pos_weight and hasattr(clf.named_steps["model"], "set_params"):
                pos = int((y_train==1).sum())
                neg = int((y_train==0).sum())
                if pos > 0:
                    clf.named_steps["model"].set_params(scale_pos_weight=neg/pos)

            clf.fit(X_train, y_train)

            # predictions
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf.named_steps["model"], "predict_proba") else None
            if y_prob is not None:
                y_pred_thr = (y_prob >= decision_threshold).astype(int)
            else:
                y_pred_thr = y_pred

            acc = accuracy_score(y_test, y_pred_thr)
            prec = precision_score(y_test, y_pred_thr, zero_division=0)
            rec = recall_score(y_test, y_pred_thr, zero_division=0)
            f1 = f1_score(y_test, y_pred_thr, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan

            _store_model_to_session(clf, selected_features, target)
            trained = True

            k1,k2,k3,k4,k5 = st.columns(5)
            k1.metric("Accuracy", f"{acc:.3f}")
            k2.metric("Precision", f"{prec:.3f}")
            k3.metric("Recall", f"{rec:.3f}")
            k4.metric("F1", f"{f1:.3f}")
            k5.metric("ROC AUC", f"{roc_auc:.3f}" if not np.isnan(roc_auc) else "—")
            st.caption(f"Threshold klasifikasi saat ini: **{decision_threshold:.2f}**")

            # Confusion Matrix (pakai threshold)
            cm = confusion_matrix(y_test, y_pred_thr)
            cm_fig = px.imshow(cm, text_auto=True, aspect="equal",
                               title="Confusion Matrix",
                               labels=dict(x="Predicted", y="Actual"),
                               color_continuous_scale="Plasma")
            st.plotly_chart(cm_fig, use_container_width=True)

            # ROC Curve
            if y_prob is not None:
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_fig = go.Figure()
                roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
                roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(dash="dash")))
                roc_fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                st.plotly_chart(roc_fig, use_container_width=True)

            # Feature Importance (Permutation)
            with st.expander("🧠 Feature Importance (Permutation)", expanded=False):
                try:
                    r = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=random_state, n_jobs=-1)
                    try:
                        feat_names = clf.named_steps["preprocess"].get_feature_names_out()
                    except Exception:
                        encoder = clf.named_steps["preprocess"].named_transformers_["cat"].named_steps["encoder"]
                        cat_features = encoder.get_feature_names_out(raw_cat_cols).tolist()
                        feat_names = np.array(num_cols + cat_features, dtype=object)
                    n_imp = len(r.importances_mean)
                    feat_names = np.array(feat_names[:n_imp], dtype=object)
                    importances = pd.DataFrame({
                        "feature": feat_names,
                        "importance": r.importances_mean
                    }).sort_values("importance", ascending=False).head(30)
                    st.plotly_chart(px.bar(importances, x="importance", y="feature", orientation="h", title="Top Features", template="plotly_dark"), use_container_width=True)
                    st.dataframe(importances, use_container_width=True)
                except Exception as e:
                    st.warning(f"Gagal menghitung permutation importance: {e}")

            # Save model
            buffer = BytesIO()
            pickle.dump({"pipeline": clf, "features": selected_features, "target": target}, buffer)
            st.download_button("💾 Download Model (.pkl)", data=buffer.getvalue(), file_name="default_model.pkl", mime="application/octet-stream")

            st.success("Model siap dipakai untuk prediksi! Lanjut ke panel **Prediksi Individual** di bawah.")
        except Exception as e:
            st.error(f"Gagal melatih model: {e}")

# ----------------- PREDICTION FORM -----------------
st.markdown("### 🔮 Prediksi Individual")
with st.expander("Isi fitur pelanggan lalu klik Prediksi", expanded=True):
    pred_cols = st.columns(min(3, max(1, len(selected_features))))
    input_dict = {}
    for i, col_name in enumerate(selected_features):
        with pred_cols[i % len(pred_cols)]:
            if col_name in raw_cat_cols:
                choices = sorted(df[col_name].dropna().astype(str).unique().tolist())
                val = st.selectbox(col_name, options=choices, key=f"pred_{col_name}")
                input_dict[col_name] = val
            elif col_name in num_cols:
                med = float(pd.to_numeric(df[col_name], errors="coerce").dropna().median()) if pd.to_numeric(df[col_name], errors="coerce").notna().any() else 0.0
                val = st.number_input(col_name, value=med, step=1.0, key=f"pred_{col_name}")
                input_dict[col_name] = val
            else:
                val = st.text_input(col_name, key=f"pred_{col_name}")
                input_dict[col_name] = val

    if st.button("🔮 Prediksi Default Sekarang"):
        clf_sess = _get_model_from_session()
        if clf_sess is None:
            st.warning("Latih model dulu di atas, atau unggah model yang sudah dilatih.")
        else:
            try:
                X_new = pd.DataFrame([input_dict])
                if hasattr(clf_sess.named_steps["model"], "predict_proba"):
                    proba = clf_sess.predict_proba(X_new)[:, 1][0]
                    pred_label = int(proba >= decision_threshold)
                else:
                    pred_label = int(clf_sess.predict(X_new)[0])
                    proba = None

                left, right = st.columns([0.5, 0.5])
                with left:
                    st.metric("Hasil Prediksi", "Default" if pred_label==1 else "Tidak Default")
                with right:
                    if proba is not None:
                        st.metric("Probabilitas Default", f"{proba*100:.1f}%")
                    else:
                        st.caption("Model tidak mendukung probabilitas.")
            except Exception as e:
                st.error(f"Gagal memprediksi: {e}")

# ----------------- EDA QUICK LOOK -----------------
st.markdown("### 🌈 EDA Singkat & Berwarna")
eda1, eda2, eda3 = st.columns(3)
with eda1:
    if target in df.columns and set(pd.Series(df[target]).dropna().unique()).issubset({0,1}):
        rate = df[target].mean()
        st.metric("Default Rate", f"{rate*100:.1f}%")
with eda2:
    if len(num_cols) > 0:
        num_pick = st.selectbox("Distribusi Numerik", options=num_cols, index=0)
        st.plotly_chart(px.histogram(df, x=num_pick, color=target if target in df.columns else None, nbins=40, template="plotly_dark", title=f"Distribusi {num_pick}"), use_container_width=True)
with eda3:
    if len(cat_cols) > 0:
        cat_pick = st.selectbox("Proporsi Kategorikal", options=cat_cols, index=0)
        cat_counts = (
            df[cat_pick]
            .astype(str)
            .value_counts(dropna=False)
            .rename_axis(cat_pick)
            .reset_index(name="Count")
        )
        st.plotly_chart(
            px.bar(cat_counts, x=cat_pick, y="Count", title=f"Count {cat_pick}", template="plotly_dark"),
            use_container_width=True
        )

st.markdown("---")
st.markdown("<center><small>Dibuat dengan ❤️ untuk portofolio loan default prediction. Siap dipush ke GitHub & deploy di Streamlit Cloud.</small></center>", unsafe_allow_html=True)
