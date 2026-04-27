import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

model = joblib.load("best_model.pkl")

st.title("Student Exam Score Predictor")
st.write("This project predicts student exam scores using Machine Learning based on study habits, attendance, sleep, and mental health.")

study_hours = st.slider("Study Hours per Day", 0.0, 12.0 ,2.0)
attendance = st.slider("Attendance Precentage", 0.0, 100.0, 80.0)
mental_health = st.slider("Mental Health Rating (1-10)", 1, 10, 5)
sleep_hours = st.slider("Sleep Hours per Night", 0.0, 12.0, 7.0)
part_time_job = st.selectbox("Part-Time Job", ["No", "Yes"] )

ptj_encoded = 1 if part_time_job == "Yes" else 0

if study_hours < 2:
    st.warning("⚠️ Very low study hours may result in low score!")

if st.button("Predict Exam Score"):
    input_data = np.array([[study_hours, attendance, mental_health, sleep_hours,ptj_encoded]])
    st.subheader("Your Inputs")
    st.write({
    "Study Hours": study_hours,
    "Attendance": attendance,
    "Mental Health": mental_health,
    "Sleep Hours": sleep_hours,
    "Part Time Job": part_time_job
})
    prediction = model.predict(input_data)[0]

    prediction = max(0, min(200,prediction))
    st.write("Model used: Linear Regression")
    st.success(f"🎯 Predicted Exam Score: {prediction:.2f}")

    hours_range = np.linspace(0, 12, 50)

    predictions = []
    for h in hours_range:
       temp_input = np.array([[h, attendance, mental_health, sleep_hours, ptj_encoded]])
       pred = model.predict(temp_input)[0]
       predictions.append(pred)

    fig, ax = plt.subplots()
    ax.plot(hours_range, predictions)   # line graph
    ax.scatter(study_hours, prediction, color='red')  # your point

    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Predicted Score")
    ax.set_title("Study Hours vs Predicted Score")

    ax.grid(True)

    st.pyplot(fig)


st.markdown("---")
st.write("Made by Tarushi Modi | Machine Learning Project")
    
