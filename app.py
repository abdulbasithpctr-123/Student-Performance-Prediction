
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background-color: #111827;
}
.metric-card {
    background: linear-gradient(145deg, #192538, #111827);
    border: 1px solid #26364e;
    padding: 25px;
    border-radius: 18px;
    min-height: 125px;
}
.metric-label {
    color: #94a3b8;
    font-size: 14px;
    font-weight: 600;
}
.metric-value {
    color: #2dd4bf;
    font-size: 28px;
    font-weight: 800;
    margin-top: 12px;
}
h1, h2, h3 {
    color: #f8fafc !important;
}
.stButton > button {
    background-color: #14b8a6;
    color: #07111f;
    font-weight: bold;
    border-radius: 10px;
    border: none;
    padding: 12px 25px;
}
.stButton > button:hover {
    background-color: #2dd4bf;
    color: #07111f;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
    model_available = True
except Exception as error:
    model_available = False
    model_error = str(error)


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🎓 Student AI")
    st.caption("PERFORMANCE INTELLIGENCE")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "👨‍🎓 Student Prediction",
            "📊 Performance Analytics",
            "ℹ️ About Project"
        ],
        label_visibility="visible"
    )

    st.divider()
    st.caption("POWERED BY MACHINE LEARNING")
    st.caption("Ridge Regression Model")


# --------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------

if page == "🏠 Dashboard":

    st.title("Student Performance Dashboard")
    st.write(
        "AI-powered academic performance prediction and student analytics"
    )

    col1, col2, col3, col4 = st.columns(4)

    metrics = [
        ("MODEL", "Ridge Regression"),
        ("FEATURES", "19"),
        ("MODEL R²", "72.32%"),
        ("CROSS-VAL MAE", "0.507")
    ]

    for column, (label, value) in zip(
        [col1, col2, col3, col4], metrics
    ):
        with column:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### Welcome to Student Performance AI")

    st.write(
        "This application uses a trained Machine Learning model "
        "to estimate student examination scores based on academic, "
        "personal and educational factors."
    )

    st.write(
        "Navigate to Student Prediction to enter student information "
        "and generate a predicted examination score."
    )

    if model_available:
        st.success("Trained prediction model loaded successfully.")
    else:
        st.error(f"Unable to load model: {model_error}")


# --------------------------------------------------
# STUDENT PREDICTION PAGE
# --------------------------------------------------

elif page == "👨‍🎓 Student Prediction":

    st.title("Student Performance Prediction")
    st.write(
        "Enter the student's details below to estimate their examination score."
    )

    if not model_available:
        st.error(f"Model could not be loaded: {model_error}")
        st.stop()

    with st.form("prediction_form"):

        st.subheader("📚 Academic Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            hours_studied = st.number_input(
                "Hours Studied per Week",
                min_value=1,
                max_value=44,
                value=20
            )

        with col2:
            attendance = st.number_input(
                "Attendance (%)",
                min_value=60,
                max_value=100,
                value=80
            )

        with col3:
            previous_scores = st.number_input(
                "Previous Scores",
                min_value=50,
                max_value=100,
                value=75
            )

        col4, col5, col6 = st.columns(3)

        with col4:
            tutoring_sessions = st.number_input(
                "Tutoring Sessions",
                min_value=0,
                max_value=10,
                value=1
            )

        with col5:
            sleep_hours = st.number_input(
                "Sleep Hours",
                min_value=4,
                max_value=10,
                value=7
            )

        with col6:
            physical_activity = st.number_input(
                "Physical Activity (hours)",
                min_value=0,
                max_value=10,
                value=3
            )

        st.divider()
        st.subheader("🏫 Educational Factors")

        col1, col2, col3 = st.columns(3)

        with col1:
            parental_involvement = st.selectbox(
                "Parental Involvement",
                ["Low", "Medium", "High"]
            )

            access_to_resources = st.selectbox(
                "Access to Resources",
                ["Low", "Medium", "High"]
            )

            extracurricular = st.selectbox(
                "Extracurricular Activities",
                ["Yes", "No"]
            )

            motivation = st.selectbox(
                "Motivation Level",
                ["Low", "Medium", "High"]
            )

            internet_access = st.selectbox(
                "Internet Access",
                ["Yes", "No"]
            )

        with col2:
            family_income = st.selectbox(
                "Family Income",
                ["Low", "Medium", "High"]
            )

            teacher_quality = st.selectbox(
                "Teacher Quality",
                ["Low", "Medium", "High"]
            )

            school_type = st.selectbox(
                "School Type",
                ["Public", "Private"]
            )

            peer_influence = st.selectbox(
                "Peer Influence",
                ["Positive", "Neutral", "Negative"]
            )

        with col3:
            learning_disabilities = st.selectbox(
                "Learning Disabilities",
                ["Yes", "No"]
            )

            parental_education = st.selectbox(
                "Parental Education Level",
                ["High School", "College", "Postgraduate"]
            )

            distance_from_home = st.selectbox(
                "Distance from Home",
                ["Near", "Moderate", "Far"]
            )

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

        st.divider()

        submitted = st.form_submit_button(
            "🚀 Predict Exam Score",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame([{
            "Hours_Studied": hours_studied,
            "Attendance": attendance,
            "Parental_Involvement": parental_involvement,
            "Access_to_Resources": access_to_resources,
            "Extracurricular_Activities": extracurricular,
            "Sleep_Hours": sleep_hours,
            "Previous_Scores": previous_scores,
            "Motivation_Level": motivation,
            "Internet_Access": internet_access,
            "Tutoring_Sessions": tutoring_sessions,
            "Family_Income": family_income,
            "Teacher_Quality": teacher_quality,
            "School_Type": school_type,
            "Peer_Influence": peer_influence,
            "Physical_Activity": physical_activity,
            "Learning_Disabilities": learning_disabilities,
            "Parental_Education_Level": parental_education,
            "Distance_from_Home": distance_from_home,
            "Gender": gender
        }])

        try:
            prediction = float(model.predict(input_data)[0])

            st.divider()
            st.subheader("🎯 Prediction Result")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric(
                    "Predicted Exam Score",
                    f"{prediction:.2f}"
                )

            with result_col2:
                st.metric(
                    "Predicted Percentage",
                    f"{prediction:.2f}%"
                )

            st.progress(
                max(0.0, min(1.0, prediction / 100))
            )

            if prediction >= 80:
                st.success("Excellent predicted academic performance!")
            elif prediction >= 70:
                st.info("Good predicted academic performance.")
            elif prediction >= 60:
                st.warning("Moderate predicted academic performance.")
            else:
                st.error("The predicted score indicates that additional academic support may be helpful.")

            st.caption(
                "This is a machine learning estimate, not an actual examination result."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# PERFORMANCE ANALYTICS PAGE
# --------------------------------------------------

elif page == "📊 Performance Analytics":

    st.title("Performance Analytics")

    st.write(
        "Explore the important findings from the model development process."
    )

    st.subheader("Model Evaluation")

    col1, col2, col3 = st.columns(3)

    col1.metric("R² Score", "76.97%", help="Previously reported test-set result")
    col2.metric("MAE", "0.4524")
    col3.metric("RMSE", "1.8041")

    st.info(
        "These are the previously recorded test-set evaluation results. "
        "They are not recalculated by this page."
    )

    st.subheader("Important Features")

    importance_data = pd.DataFrame({
        "Feature": [
            "Attendance",
            "Hours Studied",
            "Previous Scores",
            "Access to Resources",
            "Parental Involvement"
        ],
        "Mutual Information": [
            0.318968,
            0.171761,
            0.020749,
            0.033008,
            0.013490
        ]
    }).sort_values("Mutual Information", ascending=False)

    st.bar_chart(
        importance_data.set_index("Feature")
    )

    st.caption(
        "Feature values are mutual information scores from the project analysis."
    )


# --------------------------------------------------
# ABOUT PROJECT PAGE
# --------------------------------------------------

elif page == "ℹ️ About Project":

    st.title("About the Project")

    st.markdown("""
    ### Student Performance Prediction System

    This project uses Machine Learning to estimate student examination
    scores based on academic, personal and educational factors.

    **Dataset**
    - 6,607 student records
    - 20 original columns
    - 19 input features
    - 1 prediction target: Exam_Score

    **Machine Learning**
    - Linear Regression
    - Ridge Regression
    - Random Forest
    - Gradient Boosting
    - Decision Tree
    - ElasticNet
    - Support Vector Regression

    **Selected Model**
    - Ridge Regression
    - Optimized alpha: 10
    - Five-fold cross-validation

    **Technologies**
    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Streamlit
    - Plotly
    """)

    st.info(
        "The system is intended for academic analysis and decision support. "
        "Predictions should not be treated as guaranteed examination results."
    )
