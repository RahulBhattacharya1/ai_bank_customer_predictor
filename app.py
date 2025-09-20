import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
import joblib
import streamlit as st

@st.cache_resource
def load_pipeline():
    root = Path(__file__).parent
    candidates = [
        root / "models" / "subscription_pipeline.joblib",  # preferred
        root / "subscription_pipeline.joblib",             # fallback (root)
    ]
    for p in candidates:
        if p.exists():
            try:
                return joblib.load(p)
            except Exception as e:
                st.error(f"Found model at {p}, but failed to load: {e}")
                st.stop()

    # If we get here, nothing was found. Show what *is* in the workspace to help debug.
    found_joblibs = list(root.rglob("*.joblib"))
    st.error(
        "Model file not found. Expected `models/subscription_pipeline.joblib`.\n\n"
        f"Detected joblib files: {found_joblibs if found_joblibs else 'None'}"
    )
    st.info("Fix: Upload `subscription_pipeline.joblib` into a `models/` folder at your repo root.")
    st.stop()

pipe = load_pipeline()


st.title("Customer Subscription Predictor")
st.write("Predict the probability that a customer will subscribe before contacting them.")

# --- Build input widgets for key features (exclude 'id' and 'duration') ---
# You can adjust the defaults/choices to your data distribution.
age = st.number_input("Age", min_value=17, max_value=100, value=35, step=1)
balance = st.number_input("Balance", value=500, step=50)

job = st.selectbox("Job", [
    "admin.", "technician", "services", "blue-collar", "entrepreneur",
    "housemaid", "management", "retired", "self-employed", "student",
    "unemployed", "unknown"
])

marital = st.selectbox("Marital", ["single", "married", "divorced", "unknown"])
education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
default = st.selectbox("Default on Credit", ["no", "yes", "unknown"])
housing = st.selectbox("Housing Loan", ["no", "yes", "unknown"])
loan = st.selectbox("Personal Loan", ["no", "yes", "unknown"])
contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])

day = st.number_input("Last Contact Day of Month", min_value=1, max_value=31, value=15, step=1)
month = st.selectbox("Last Contact Month", [
    "jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"
])

campaign = st.number_input("Contacts During This Campaign", min_value=1, value=1, step=1)
pdays = st.number_input("Days Passed After Last Contact (pdays)", value=999, step=1)
previous = st.number_input("Number of Contacts Before This Campaign", min_value=0, value=0, step=1)
poutcome = st.selectbox("Previous Campaign Outcome", ["success","failure","other","unknown"])

# Construct a single-row DataFrame that matches training columns (except dropped ones)
row = pd.DataFrame([{
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "day": day,
    "month": month,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome
}])

st.subheader("Your Input")
st.dataframe(row, use_container_width=True)

# Optional: decision threshold slider (default 0.5)
thr = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01)

if st.button("Predict"):
    # use pipeline directly; it includes preprocessing and model
    prob = float(pipe.predict_proba(row)[0, 1])
    pred = int(prob >= thr)

    st.markdown(f"**Probability of subscription:** {prob:.3f}")
    if pred == 1:
        st.success("Predicted: Subscribe (1)")
    else:
        st.info("Predicted: Not Subscribe (0)")

    # Basic explanation via top coefficients for logistic regression
    # NOTE: This is a lightweight, heuristic peek (not full SHAP).
    try:
        model = pipe.named_steps["model"]
        st.caption("Model type: Logistic Regression")
    except Exception:
        pass
