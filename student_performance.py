import os
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
PYTHON_VERSION=3.11.11
# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Performance & Prediction Hub",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Student Performance Analytics & Prediction Hub")
st.markdown(
    "Explore dataset trends, visualize distributions, and predict individual student pass outcomes."
)

# ---------------------------------------------------------
# 2. DATA LOADING & DATA CLEANING
# ---------------------------------------------------------
file_path = "student_performance_prediction.csv"

st.sidebar.header("⚙️ Data Settings & Inputs")
uploaded_file = st.sidebar.file_uploader("Upload CSV File Manually", type=["csv"])


@st.cache_data
def load_and_clean_data(path, uploader):
    try:
        if uploader is not None:
            df = pd.read_csv(uploader)
        elif os.path.exists(path):
            df = pd.read_csv(path)
        else:
            return None

        # Clean column headers
        df.columns = df.columns.str.strip()

        # Map to internal columns
        rename_dict = {
            "Student ID": "Name",
            "Attendance Rate": "Attendance",
            "Previous Grades": "Final",
            "Passed": "Result",
        }
        df = df.rename(columns=rename_dict)

        # Standardize 'Result' target column and handle string 'Nan'
        if "Result" in df.columns:
            df["Result"] = df["Result"].astype(str).str.strip().str.capitalize()
            # Drop invalid/unknown target rows
            df = df[df["Result"].isin(["Yes", "No", "Pass", "Fail"])]

        # Clean numerical bounds & noise
        if "Attendance" in df.columns:
            df["Attendance"] = df["Attendance"].clip(lower=0, upper=100)
        if "Study Hours per Week" in df.columns:
            df["Study Hours per Week"] = df["Study Hours per Week"].clip(lower=0, upper=80)
        if "Final" in df.columns:
            df["Final"] = df["Final"].clip(lower=0, upper=100)

        # Drop remaining rows with blank features
        df = df.dropna()

        return df
    except Exception as e:
        st.error(f"Error loading CSV data: {e}")
        return None


df = load_and_clean_data(file_path, uploaded_file)

if df is None or df.empty:
    st.error("⚠️ **Dataset invalid or missing!**")
    st.info(
        f"Looking for: **'{file_path}'** in current folder:\n`{os.getcwd()}`"
    )
    st.warning("👉 Upload your CSV via the sidebar to load the app.")
    st.stop()

# ---------------------------------------------------------
# 3. MACHINE LEARNING MODEL TRAINING
# ---------------------------------------------------------


@st.cache_resource
def train_model(data):
    train_data = data.copy()

    feature_cols = [
        "Study Hours per Week",
        "Attendance",
        "Final",
        "Participation in Extracurricular Activities",
        "Parent Education Level",
    ]

    target_col = "Result"

    # Encode categorical columns
    encoders = {}
    for col in ["Participation in Extracurricular Activities", "Parent Education Level"]:
        le = LabelEncoder()
        train_data[col] = le.fit_transform(train_data[col].astype(str))
        encoders[col] = le

    target_le = LabelEncoder()
    train_data[target_col] = target_le.fit_transform(train_data[target_col].astype(str))

    X = train_data[feature_cols]
    y = train_data[target_col]

    # Fast Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    return clf, encoders, target_le, feature_cols


model, encoders, target_encoder, feature_cols = train_model(df)

# ---------------------------------------------------------
# 4. TAB NAVIGATION
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["🔮 Predict Student Result", "📊 Dataset Analytics Dashboard"])

# TAB 1: PREDICTION FORM
with tab1:
    st.subheader("📝 Single Student Feature Input")
    st.markdown("Enter student parameters below and click predict:")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            study_hours = st.number_input(
                "Study Hours per Week", min_value=0.0, max_value=80.0, value=12.0, step=0.5
            )
            attendance = st.number_input(
                "Attendance Rate (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0
            )

        with c2:
            previous_grade = st.number_input(
                "Previous Grade / Score", min_value=0.0, max_value=100.0, value=65.0, step=1.0
            )
            extracurricular = st.selectbox(
                "Extracurricular Activities",
                options=list(encoders["Participation in Extracurricular Activities"].classes_),
            )

        with c3:
            parent_edu = st.selectbox(
                "Parent Education Level",
                options=list(encoders["Parent Education Level"].classes_),
            )

        submit_btn = st.form_submit_button("🔮 Predict Result", use_container_width=True)

    if submit_btn:
        encoded_extra = encoders["Participation in Extracurricular Activities"].transform(
            [extracurricular]
        )[0]
        encoded_parent = encoders["Parent Education Level"].transform([parent_edu])[0]

        input_df = pd.DataFrame(
            [[study_hours, attendance, previous_grade, encoded_extra, encoded_parent]],
            columns=feature_cols,
        )

        pred_numeric = model.predict(input_df)[0]
        pred_label = target_encoder.inverse_transform([pred_numeric])[0]
        probs = model.predict_proba(input_df)[0]

        st.markdown("---")
        st.subheader("🎯 Outcome Prediction")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            if pred_label.capitalize() in ["Yes", "Pass"]:
                st.success("🎉 **Predicted Result: PASSED**")
            else:
                st.error("⚠️ **Predicted Result: FAILED**")

        with res_col2:
            pass_idx = list(target_encoder.classes_).index(pred_label)
            confidence = round(probs[pass_idx] * 100, 2)
            st.metric("Model Confidence", f"{confidence}%")

# TAB 2: DASHBOARD
with tab2:
    st.subheader("📊 Dataset Analysis")

    status_options = ["All"] + list(df["Result"].unique())
    selected_status = st.selectbox("Filter Data by Result", status_options)

    filtered_df = df if selected_status == "All" else df[df["Result"] == selected_status]

    if st.button("🔍 Run Dataset Analysis", type="primary"):
        # Metrics
        st.markdown("#### 📌 Key Performance Metrics")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Cleaned Students", f"{len(filtered_df):,}")
        k2.metric("Average Score", f"{round(filtered_df['Final'].mean(), 2)}")
        k3.metric("Avg Study Hours/Wk", f"{round(filtered_df['Study Hours per Week'].mean(), 2)}")
        k4.metric("Avg Attendance Rate", f"{round(filtered_df['Attendance'].mean(), 2)}%")

        st.markdown("---")

        # Visualizations (No plt.show calls!)
        st.markdown("#### 📈 Visual Plots")
        r1_c1, r1_c2 = st.columns(2)

        with r1_c1:
            st.markdown("**Attendance Rate vs. Final Score**")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.scatter(
                filtered_df["Attendance"],
                filtered_df["Final"],
                color="#1f77b4",
                alpha=0.3,
                s=12,
            )
            ax1.set_xlabel("Attendance Rate (%)")
            ax1.set_ylabel("Score")
            ax1.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig1)

        with r1_c2:
            st.markdown("**Study Hours vs. Final Score**")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.scatter(
                filtered_df["Study Hours per Week"],
                filtered_df["Final"],
                color="#9467bd",
                alpha=0.3,
                s=12,
            )
            ax2.set_xlabel("Study Hours per Week")
            ax2.set_ylabel("Score")
            ax2.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig2)

        r2_c1, r2_c2 = st.columns(2)

        with r2_c1:
            st.markdown("**Sample Scores (First 20 Students)**")
            sample_df = filtered_df.head(20)
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            ax3.bar(
                sample_df["Name"].astype(str),
                sample_df["Final"],
                color="#2ca02c",
                edgecolor="black",
            )
            ax3.set_xlabel("Student ID")
            ax3.set_ylabel("Score")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig3)

        with r2_c2:
            st.markdown("**Pass vs. Fail Distribution**")
            counts = filtered_df["Result"].value_counts()
            fig4, ax4 = plt.subplots(figsize=(5, 4))
            ax4.pie(
                counts,
                labels=counts.index,
                autopct="%1.1f%%",
                startangle=90,
                colors=["#4CAF50", "#FF5252"],
            )
            st.pyplot(fig4)

        st.markdown("---")
        st.markdown("#### 📋 Cleaned Dataset Explorer")
        st.dataframe(filtered_df, use_container_width=True)
