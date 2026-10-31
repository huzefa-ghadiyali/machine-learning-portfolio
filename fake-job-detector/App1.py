import joblib
model = joblib.load(r"W:\Storage\Users\Huzefa\Work\Swotting\Jupyter Notebook\Machine Learning Jupyter notebook\End-To-End Projects\Fake Job Detector\model.joblib")


def predict_job(description):
    description = description.strip()
    
    prediction = model.predict([description])
    
    decision_score = float(model.decision_function([description])[0])

    return {
        "prediction": prediction,
        "decision_score": decision_score
    }


import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Fake Job Posting Detector",
    page_icon="🔎",
    layout="centered"
)

# Header
st.title("🔎 Fake Job Posting Detector")

st.write(
    "Analyze a job posting using machine learning "
    "to identify potentially fraudulent listings."
)

st.info(
    "Paste the job description below. The model's "
    "prediction is an estimate, not proof of fraud."
)

# User input
description = st.text_area(
    "Job Description",
    placeholder="Paste the job title, description, requirements, "
                "and other available details here...",
    height=250,
    max_chars=20000
)

# Prediction
if st.button("Analyze Job Posting", type="primary",
             use_container_width=True):

    if not description.strip():
        st.warning("Please enter a job description.")

    elif len(description.strip()) < 30:
        st.warning(
            "Please provide more details "
            "(at least 30 characters)."
        )

    else:
        try:
            with st.spinner("Analyzing job posting..."):
                result = predict_job(description)

            prediction = result["prediction"]
            decision_score = result["decision_score"]

            st.divider()
            st.subheader("Analysis Results")

            if prediction == 1:
                st.error(
                    "⚠️ Potentially Fraudulent Job Posting"
                )
                st.write(
                    "The model classified this posting "
                    "as potentially fraudulent."
                )

            else:
                st.success(
                    "✅ Predicted to Be Legitimate"
                )
                st.write(
                    "The model classified this posting "
                    "as likely legitimate."
                )

            st.metric(
                label="Model Decision Score",
                value=f"{decision_score:.3f}"
            )

            st.caption(
                "The decision score indicates which class "
                "the model favors. It is not a probability "
                "or a percentage chance of fraud."
            )

            with st.expander("What does this result mean?"):
                st.write(
                    "- A positive decision score generally "
                    "favors the fraudulent class (label 1).\n"
                    "- A negative decision score generally "
                    "favors the legitimate class (label 0).\n"
                    "- Scores closer to zero indicate that "
                    "the prediction is closer to the model's "
                    "decision boundary.\n"
                    "- Always verify the employer, company "
                    "website, contact details, and requests "
                    "for payment independently."
                )

        except ValueError as e:
            st.warning(str(e))

        except Exception as e:
            st.error(
                f"An unexpected error occurred: {e}"
            )

# Footer
st.divider()

st.caption(
    "Built with Python, Streamlit, scikit-learn, "
    "TF-IDF and LinearSVC."
)