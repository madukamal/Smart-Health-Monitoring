"""
Smart Health Monitoring System for Water-Borne Disease Risk Categorization
Marwadi University - Student Project

Three-page Streamlit website:
    1. Home                    - project overview and dataset summary
    2. Water & Health Input    - form for water quality and patient symptoms
    3. Risk Prediction         - predicted risk category + risk-distribution chart

Run from the project's app/ folder:
    streamlit run dashboard.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths (relative to this file, so the app works from any working directory)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "disease_risk_model.pkl"
DATA_PATH = BASE_DIR / "dataset" / "water_health_data.csv"

RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}
RISK_ORDER = ["Low", "Medium", "High"]
RISK_COLORS = {"Low": "#2e7d32", "Medium": "#f9a825", "High": "#c62828"}

FEATURE_ORDER = [
    "Age", "Gender", "pH", "Turbidity", "Temperature",
    "Fever", "Diarrhea", "Vomiting", "Abdominal_Pain",
]

PAGES = ["🏠 Home", "📝 Water & Health Input", "🔍 Risk Prediction"]

DISCLAIMER = (
    "**Experimental prediction.** This model was trained on only 12 synthetic "
    "records, so its output is for demonstration purposes only. It is not a "
    "medical diagnosis and must not be used for real health decisions."
)

DEFAULT_INPUTS = {
    "age": 25,
    "gender": "Male",
    "ph": 7.0,
    "turbidity": 2.0,
    "temperature": 25.0,
    "fever": False,
    "diarrhea": False,
    "vomiting": False,
    "abdominal_pain": False,
}

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Health Monitoring",
    page_icon="💧",
    layout="wide",
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def init_state():
    for key, value in DEFAULT_INPUTS.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("submitted", False)
    st.session_state.setdefault("page", PAGES[0])


def go_to(page):
    st.session_state["page"] = page


def submit_form():
    """Runs as the form's submit callback: save the form values, then open the result page."""
    s = st.session_state
    for key in DEFAULT_INPUTS:
        s[key] = s["w_" + key]
    s["submitted"] = True
    s["page"] = PAGES[2]


def build_input_row():
    """Turn the stored form values into the one-row DataFrame the model expects."""
    s = st.session_state
    row = {
        "Age": int(s["age"]),
        "Gender": 0 if s["gender"] == "Male" else 1,
        "pH": float(s["ph"]),
        "Turbidity": float(s["turbidity"]),
        "Temperature": float(s["temperature"]),
        "Fever": int(s["fever"]),
        "Diarrhea": int(s["diarrhea"]),
        "Vomiting": int(s["vomiting"]),
        "Abdominal_Pain": int(s["abdominal_pain"]),
    }
    return pd.DataFrame([row])[FEATURE_ORDER]


def water_quality_notes(ph, turbidity, temperature):
    """Simple rule-of-thumb comments shown next to the prediction."""
    notes = []
    if ph < 6.5 or ph > 8.5:
        notes.append(f"pH {ph:.1f} is outside the usual drinking-water range (6.5 - 8.5).")
    if turbidity > 5:
        notes.append(f"Turbidity {turbidity:.1f} NTU is above the common 5 NTU guideline limit.")
    if temperature > 30:
        notes.append(f"Water temperature {temperature:.0f} °C is warm, which favours bacterial growth.")
    if not notes:
        notes.append("Water readings are within the common reference ranges.")
    return notes


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_home():
    st.title("💧 Smart Health Monitoring System")
    st.subheader("Water-Borne Disease Risk Categorization Using Machine Learning")
    st.caption("Marwadi University - Student Project")

    st.warning(DISCLAIMER)

    st.markdown(
        """
        Water-borne diseases such as cholera, typhoid and dysentery spread through
        contaminated water. This system combines **water-quality readings** with
        **patient symptoms** and uses a Random Forest classifier to place a case into
        one of three risk categories: **Low**, **Medium** or **High**.
        """
    )

    st.markdown("### How it works")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1. Enter data**\n\nProvide water pH, turbidity, temperature, age, gender and symptoms.")
    c2.markdown("**2. Model predicts**\n\nA trained Random Forest model estimates the risk category.")
    c3.markdown("**3. View result**\n\nSee the predicted category and a risk-distribution chart.")

    st.button(
        "Start a prediction →",
        type="primary",
        on_click=go_to,
        args=(PAGES[1],),
    )

    st.markdown("---")
    st.markdown("### Training dataset overview")
    data = load_data()

    m1, m2, m3 = st.columns(3)
    m1.metric("Records", len(data))
    m2.metric("Features", len(data.columns) - 1)
    m3.metric("Risk categories", data["Risk"].nunique())

    left, right = st.columns(2)
    with left:
        st.markdown("**Risk category distribution in the dataset**")
        counts = data["Risk"].value_counts().reindex(RISK_ORDER, fill_value=0)
        st.bar_chart(counts)
    with right:
        st.markdown("**Dataset preview**")
        st.dataframe(data, use_container_width=True, hide_index=True)

    st.info("The dataset is small and synthetic. It exists to demonstrate the pipeline, not to make real predictions.")


def page_input():
    st.title("📝 Water & Health Input")
    st.write("Enter the water-quality readings and the patient's details, then run the prediction.")

    with st.form("input_form"):
        st.markdown("#### 🌊 Water quality")
        w1, w2, w3 = st.columns(3)
        w1.number_input("Water pH", 0.0, 14.0, float(st.session_state["ph"]), step=0.1, key="w_ph")
        w2.number_input(
            "Turbidity (NTU)", 0.0, 100.0, float(st.session_state["turbidity"]), step=0.1,
            key="w_turbidity",
        )
        w3.number_input(
            "Water temperature (°C)", 0.0, 60.0, float(st.session_state["temperature"]), step=0.5,
            key="w_temperature",
        )

        st.markdown("#### 🏥 Patient information")
        p1, p2 = st.columns(2)
        p1.number_input("Age", 1, 120, int(st.session_state["age"]), key="w_age")
        p2.selectbox(
            "Gender", ["Male", "Female"],
            index=0 if st.session_state["gender"] == "Male" else 1,
            key="w_gender",
        )

        st.markdown("**Symptoms**")
        s1, s2, s3, s4 = st.columns(4)
        s1.checkbox("Fever", value=st.session_state["fever"], key="w_fever")
        s2.checkbox("Diarrhea", value=st.session_state["diarrhea"], key="w_diarrhea")
        s3.checkbox("Vomiting", value=st.session_state["vomiting"], key="w_vomiting")
        s4.checkbox("Abdominal pain", value=st.session_state["abdominal_pain"], key="w_abdominal_pain")

        st.form_submit_button("🔍 Predict Disease Risk", type="primary", on_click=submit_form)

    st.caption("Predictions are experimental - see the note on the Home page.")


def page_prediction():
    st.title("🔍 Risk Prediction")

    if not st.session_state["submitted"]:
        st.info("No input yet. Fill in the form first.")
        st.button("Go to input form", on_click=go_to, args=(PAGES[1],))
        return

    model = load_model()
    row = build_input_row()

    predicted_class = int(model.predict(row)[0])
    risk = RISK_LABELS[predicted_class]

    # Probability of each class, mapped to Low / Medium / High.
    # Classes the model never saw in training get probability 0.
    proba = model.predict_proba(row)[0]
    prob_by_label = {label: 0.0 for label in RISK_ORDER}
    for cls, p in zip(model.classes_, proba):
        prob_by_label[RISK_LABELS[int(cls)]] = float(p)

    st.warning(DISCLAIMER)

    left, right = st.columns([1, 1.2])

    with left:
        st.markdown("### Predicted risk category")
        if risk == "High":
            st.error("🚨 HIGH RISK")
            st.write("Further investigation is recommended.")
        elif risk == "Medium":
            st.warning("⚠️ MEDIUM RISK")
            st.write("Monitor the situation and check the water source.")
        else:
            st.success("✅ LOW RISK")
            st.write("No strong indicators of water-borne disease in this input.")

        st.metric("Model confidence", f"{prob_by_label[risk] * 100:.0f}%")

        st.markdown("#### Observations")
        for note in water_quality_notes(
            st.session_state["ph"], st.session_state["turbidity"], st.session_state["temperature"]
        ):
            st.write("• " + note)
        n_symptoms = int(row[["Fever", "Diarrhea", "Vomiting", "Abdominal_Pain"]].sum(axis=1)[0])
        st.write(f"• {n_symptoms} of 4 symptoms reported.")

    with right:
        st.markdown("### Risk distribution for this input")
        chart_df = pd.DataFrame(
            {"Probability (%)": [round(prob_by_label[r] * 100, 1) for r in RISK_ORDER]},
            index=RISK_ORDER,
        )
        st.bar_chart(chart_df)
        st.caption(
            "Share of the Random Forest's trees voting for each category. "
            "With such a small training set these values are rough."
        )

    with st.expander("Input used for this prediction"):
        st.dataframe(row, use_container_width=True, hide_index=True)

    b1, b2, _ = st.columns([1, 1, 4])
    b1.button("← Edit input", on_click=go_to, args=(PAGES[1],))
    b2.button("Home", on_click=go_to, args=(PAGES[0],))


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
init_state()

st.sidebar.title("💧 Navigation")
st.sidebar.radio("Go to", PAGES, key="page")
st.sidebar.markdown("---")
st.sidebar.caption("Student prototype · synthetic data · experimental predictions")

current = st.session_state["page"]
if current == PAGES[0]:
    page_home()
elif current == PAGES[1]:
    page_input()
else:
    page_prediction()
