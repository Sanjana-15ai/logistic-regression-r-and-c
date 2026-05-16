import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score, roc_curve)

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Classification App", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    .metric-card {
        background: #f0f2f6; border-radius: 12px;
        padding: 16px 20px; text-align: center; margin-bottom: 8px;
    }
    .metric-card h2 { margin: 0; font-size: 2rem; }
    .metric-card p  { margin: 0; color: #555; font-size: 0.85rem; }
    .predict-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px; padding: 24px; color: white; text-align: center;
    }
    .predict-box h1 { margin: 0; font-size: 2.5rem; }
    .predict-box p  { margin: 4px 0 0 0; font-size: 1rem; opacity: 0.9; }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Classification App — Train & Predict")
st.markdown("Upload your CSV, pick a model, compare performance, then predict with custom inputs.")

# ── Session State ──────────────────────────────────────────────────────────────
for k in ["model", "scaler", "features", "target", "trained",
          "label_encoders", "clean_data", "results"]:
    if k not in st.session_state:
        st.session_state[k] = None
if "trained" not in st.session_state:
    st.session_state.trained = False

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("1️⃣ Upload Data")
csv_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if csv_file is None:
    st.info("👈 Upload a CSV file from the sidebar to get started.")
    st.stop()

raw_df = pd.read_csv(csv_file)

st.sidebar.header("2️⃣ Configure Columns")
all_cols = raw_df.columns.tolist()
target   = st.sidebar.selectbox("Target column (y)", all_cols, index=len(all_cols) - 1)
features = st.sidebar.multiselect("Feature columns (X)",
                                   [c for c in all_cols if c != target],
                                   default=[c for c in all_cols if c != target])

st.sidebar.header("3️⃣ Choose Model")
model_choice = st.sidebar.selectbox("Model", ["Logistic Regression", "SVM", "Random Forest"])

if model_choice == "Logistic Regression":
    C       = st.sidebar.number_input("C (regularisation)", 0.01, 100.0, 1.0)
    max_iter = st.sidebar.slider("Max Iterations", 100, 1000, 200, 50)
elif model_choice == "SVM":
    kernel  = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly"])
    C_svm   = st.sidebar.number_input("C", 0.01, 100.0, 1.0)
elif model_choice == "Random Forest":
    n_est   = st.sidebar.slider("N Estimators", 10, 300, 100, 10)
    max_dep = st.sidebar.slider("Max Depth", 1, 20, 5)

test_size  = st.sidebar.slider("Test Split", 0.1, 0.4, 0.2, 0.05)
scale_data = st.sidebar.checkbox("Scale Features", value=True)
remove_out = st.sidebar.checkbox("Remove Outliers (IQR)", value=True)

# ── Preprocessing ──────────────────────────────────────────────────────────────
df = raw_df.copy()

# Fill missing
for col in df.select_dtypes(include="number").columns:
    df[col].fillna(df[col].mean(), inplace=True)
for col in df.select_dtypes(include="object").columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Encode categoricals
label_encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Outlier removal
if remove_out:
    num_cols = df[features + [target]].select_dtypes(include="number").columns
    Q1  = df[num_cols].quantile(0.25)
    Q3  = df[num_cols].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((df[num_cols] < (Q1 - 1.5 * IQR)) | (df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    df = df[mask]

st.session_state.clean_data     = df
st.session_state.label_encoders = label_encoders

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Overview", "🏋️ Train & Evaluate", "📊 Visualise", "🎯 Predict"])

# ── Tab 1 · Data Overview ──────────────────────────────────────────────────────
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw Rows",      len(raw_df))
    c2.metric("Cleaned Rows",  len(df))
    c3.metric("Features",      len(features))
    c4.metric("Target Classes", df[target].nunique())

    st.subheader("Cleaned Data Preview")
    st.dataframe(df.head(15), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Class Distribution")
        class_counts = df[target].value_counts().reset_index()
        class_counts.columns = ["Class", "Count"]
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(class_counts["Class"].astype(str), class_counts["Count"],
               color=["#e74c3c", "#2ecc71", "#3498db"], edgecolor="white")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title("Target Distribution")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)

    with col2:
        st.subheader("Descriptive Stats")
        st.dataframe(df[features].describe(), use_container_width=True)

# ── Tab 2 · Train & Evaluate ───────────────────────────────────────────────────
with tab2:
    if not features:
        st.warning("Select at least one feature in the sidebar.")
        st.stop()

    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        X = df[features].values
        y = df[target].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y)

        scaler = None
        if scale_data:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

        if model_choice == "Logistic Regression":
            model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
        elif model_choice == "SVM":
            model = SVC(kernel=kernel, C=C_svm, probability=True, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=n_est, max_depth=max_dep, random_state=42)

        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        acc     = accuracy_score(y_test, y_pred)
        cm      = confusion_matrix(y_test, y_pred)
        report  = classification_report(y_test, y_pred, output_dict=True)
        classes = np.unique(y)
        roc_auc = None
        if len(classes) == 2:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])

        st.session_state.model    = model
        st.session_state.scaler   = scaler
        st.session_state.features = features
        st.session_state.target   = target
        st.session_state.trained  = True
        st.session_state.results  = {
            "acc": acc, "cm": cm, "report": report,
            "y_test": y_test, "y_pred": y_pred,
            "y_proba": y_proba, "classes": classes,
            "roc_auc": roc_auc, "model_name": model_choice
        }
        st.success(f"✅ {model_choice} trained successfully!")

    if st.session_state.trained:
        r = st.session_state.results

        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("✅ Accuracy",  f"{r['acc']*100:.2f}%")
        m2.metric("🏷️ Model",     r["model_name"])
        m3.metric("📐 ROC-AUC",   f"{r['roc_auc']:.4f}" if r["roc_auc"] else "N/A (multiclass)")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Confusion Matrix")
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues", ax=ax2,
                        xticklabels=r["classes"], yticklabels=r["classes"],
                        linewidths=0.5)
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("Actual")
            st.pyplot(fig2)

        with col2:
            if r["roc_auc"]:
                st.subheader("ROC Curve")
                fpr, tpr, _ = roc_curve(r["y_test"], r["y_proba"][:, 1])
                fig3, ax3 = plt.subplots(figsize=(4, 3))
                ax3.plot(fpr, tpr, color="#667eea", lw=2,
                         label=f"AUC = {r['roc_auc']:.3f}")
                ax3.plot([0, 1], [0, 1], "k--", lw=1)
                ax3.set_xlabel("False Positive Rate")
                ax3.set_ylabel("True Positive Rate")
                ax3.set_title("ROC Curve")
                ax3.legend()
                ax3.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig3)

        st.subheader("Classification Report")
        report_df = pd.DataFrame(r["report"]).transpose()
        st.dataframe(report_df.style.format(precision=3), use_container_width=True)

        # Feature importance (Random Forest only)
        if r["model_name"] == "Random Forest":
            st.subheader("🌲 Feature Importance")
            imp = pd.DataFrame({
                "Feature":   st.session_state.features,
                "Importance": st.session_state.model.feature_importances_
            }).sort_values("Importance", ascending=True)
            fig4, ax4 = plt.subplots(figsize=(5, 3))
            ax4.barh(imp["Feature"], imp["Importance"], color="#667eea")
            ax4.set_title("Feature Importance")
            ax4.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig4)

# ── Tab 3 · Visualise ──────────────────────────────────────────────────────────
with tab3:
    st.subheader("Correlation Heatmap")
    fig5, ax5 = plt.subplots(figsize=(7, 5))
    sns.heatmap(df[features + [target]].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", ax=ax5, linewidths=0.5)
    st.pyplot(fig5)

    if len(features) >= 2:
        st.subheader("Scatter Plot — Top 2 Features vs Target")
        f1 = st.selectbox("X-axis feature", features, index=0)
        f2 = st.selectbox("Y-axis feature", [f for f in features if f != f1], index=0)
        fig6, ax6 = plt.subplots(figsize=(6, 4))
        scatter = ax6.scatter(df[f1], df[f2], c=df[target],
                              cmap="RdYlGn", alpha=0.6, edgecolors="k", linewidths=0.3)
        plt.colorbar(scatter, ax=ax6, label=target)
        ax6.set_xlabel(f1)
        ax6.set_ylabel(f2)
        ax6.set_title(f"{f1} vs {f2} (coloured by {target})")
        ax6.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig6)

    st.subheader("Feature Distributions")
    sel_feat = st.selectbox("Select feature", features)
    fig7, ax7 = plt.subplots(figsize=(6, 3))
    for cls in df[target].unique():
        subset = df[df[target] == cls][sel_feat]
        ax7.hist(subset, alpha=0.6, bins=20, label=f"Class {cls}", edgecolor="white")
    ax7.set_xlabel(sel_feat)
    ax7.set_ylabel("Count")
    ax7.set_title(f"Distribution of {sel_feat} by Class")
    ax7.legend()
    ax7.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig7)

# ── Tab 4 · Predict ────────────────────────────────────────────────────────────
with tab4:
    if not st.session_state.trained:
        st.warning("⚠️ Train the model first in the **Train & Evaluate** tab.")
    else:
        st.subheader("Enter Customer Details to Predict")
        st.markdown(f"**Model:** `{st.session_state.results['model_name']}`  |  "
                    f"**Accuracy:** `{st.session_state.results['acc']*100:.2f}%`  |  "
                    f"**Target:** `{st.session_state.target}`")
        st.markdown("---")

        input_vals = {}
        feat_list  = st.session_state.features
        cols       = st.columns(min(len(feat_list), 3))

        for i, feat in enumerate(feat_list):
            col_min  = float(df[feat].min())
            col_max  = float(df[feat].max())
            col_mean = float(df[feat].mean())
            with cols[i % 3]:
                input_vals[feat] = st.number_input(
                    f"**{feat}**",
                    value=round(col_mean, 2),
                    help=f"Range: {col_min:.1f} – {col_max:.1f}"
                )

        st.markdown("---")

        if st.button("🎯 Predict Now", type="primary", use_container_width=True):
            input_arr = np.array([[input_vals[f] for f in feat_list]])

            if st.session_state.scaler:
                input_arr = st.session_state.scaler.transform(input_arr)

            prediction = st.session_state.model.predict(input_arr)[0]
            proba      = st.session_state.model.predict_proba(input_arr)[0]
            classes    = st.session_state.results["classes"]

            st.markdown("### 🧾 Prediction Result")
            r1, r2 = st.columns([1, 1])

            with r1:
                color = "#2ecc71" if prediction == 1 else "#e74c3c"
                label = "✅ Will Purchase" if prediction == 1 else "❌ Will NOT Purchase"
                st.markdown(f"""
                    <div style="background:{color}22; border-left: 5px solid {color};
                    border-radius:10px; padding:20px; text-align:center;">
                        <h2 style="color:{color}; margin:0;">{label}</h2>
                        <p style="margin:8px 0 0 0; color:#444;">
                            Predicted Class: <strong>{prediction}</strong>
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            with r2:
                st.markdown("**Confidence per Class**")
                fig8, ax8 = plt.subplots(figsize=(4, 2.5))
                bar_colors = ["#2ecc71" if c == prediction else "#e74c3c" for c in classes]
                bars = ax8.barh([f"Class {c}" for c in classes],
                                [p * 100 for p in proba],
                                color=bar_colors, edgecolor="white", height=0.4)
                ax8.set_xlim(0, 100)
                ax8.set_xlabel("Probability (%)")
                for bar, val in zip(bars, proba):
                    ax8.text(val * 100 + 1, bar.get_y() + bar.get_height() / 2,
                             f"{val*100:.1f}%", va="center", fontsize=9, fontweight="bold")
                ax8.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig8)

            st.markdown("**Your Input:**")
            st.dataframe(pd.DataFrame([input_vals]), use_container_width=True, hide_index=True)