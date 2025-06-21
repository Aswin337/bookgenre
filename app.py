import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time
import random
from datetime import datetime

# 🎨 Streamlit page setup
st.set_page_config(page_title="📚 Book Genre Preference Survey", page_icon="📖")

st.title("📚 Book Genre Preference Survey")
st.markdown("Predict your **preferred book genre** using machine learning!")

# 🧠 Load model and preprocessors
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

st.success("✅ Model and preprocessors loaded!")

# ✅ Define feature order (must match training)
feature_order = ['Gender', 'Occupation', 'Age', 'Books_Read_Per_Year']  # Update this if needed

# Separate fields
categorical_fields = list(encoders.keys())
numerical_fields = [f for f in feature_order if f not in categorical_fields]

# 🎭 Emoji map
emoji_map = {
    "Fantasy": "🧙‍♂️", "Romance": "💖", "Mystery": "🕵️‍♀️",
    "Science Fiction": "🚀", "Nonfiction": "📘", "Horror": "👻", "Comedy": "😂"
}

# ✍️ User Input UI
st.header("✍️ Enter Your Preferences")

input_data = {}
for field in categorical_fields:
    options = list(encoders[field].classes_)
    input_data[field] = st.selectbox(field, options)

for field in numerical_fields:
    input_data[field] = st.number_input(field, min_value=0.0)

submit = st.button("🎯 Predict My Book Genre")

if submit:
    st.header("🔍 Prediction Result")

    # 🔄 Encode + scale in correct order
    encoded_input = []
    for field in feature_order:
        if field in categorical_fields:
            val = encoders[field].transform([input_data[field]])[0]
        else:
            val = input_data[field]
        encoded_input.append(val)

    X_input = np.array([encoded_input])
    st.text(f"Input shape: {X_input.shape}, Expected by scaler: {scaler.n_features_in_}")

    try:
        X_scaled = scaler.transform(X_input)

        with st.spinner("🔍 Analyzing your preferences..."):
            time.sleep(2)
            pred = model.predict(X_scaled)[0]

        emoji = emoji_map.get(pred, "📚")
        st.success(f"{emoji} Your Preferred Book Genre is: **{pred}** {emoji}")
        st.balloons()
        st.snow()

        # 📖 Quote of the day
        quotes = [
            "“A reader lives a thousand lives before he dies.” – George R.R. Martin",
            "“So many books, so little time.” – Frank Zappa",
            "“Books are a uniquely portable magic.” – Stephen King",
            "“Until I feared I would lose it, I never loved to read.” – Harper Lee"
        ]
        st.info(f"📖 **Quote of the Day:** {random.choice(quotes)}")

        # 🔁 Try again
        if st.button("🔁 Try Again"):
            st.experimental_rerun()

        # 😄 Feedback
        st.header("😄 How do you feel about your result?")
        feedback = st.radio("React with an emoji!", [
            "😍 Loved it!", "🙂 It's okay", "😐 Meh", "😕 Not accurate", "😡 Hate it!"
        ])

        if feedback:
            st.success(f"Thanks for your feedback! {feedback}")
            log_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "predicted_genre": pred,
                "feedback": feedback
            }
            log_data.update(input_data)

            df_log = pd.DataFrame([log_data])
            if os.path.exists("feedback_log.csv"):
                df_log.to_csv("feedback_log.csv", mode='a', header=False, index=False)
            else:
                df_log.to_csv("feedback_log.csv", index=False)

            st.info("📁 Your feedback has been saved!")

    except Exception as e:
        st.error(f"❌ An error occurred during prediction:\n\n{e}")
