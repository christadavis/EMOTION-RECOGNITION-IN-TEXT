import streamlit as st
from transformers import pipeline

# Load the Emotion Recognition model
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

emotion_recognizer = load_model()

# Streamlit App Title
st.title("Emotion Recognition in Text")

# Input text box
user_input = st.text_area("Enter your text here:", height=200)

# Button to analyze emotion
if st.button("Analyze Emotion"):
    if user_input.strip():
        # Analyze emotion
        result = emotion_recognizer(user_input)
        emotion = result[0]['label']
        score = result[0]['score']

        # Display the results
        st.write(f"**Emotion:** {emotion}")
        st.write(f"**Confidence Score:** {score:.2f}")
    else:
        st.error("Please enter some text to analyze.")

# Footer
st.markdown("---")

