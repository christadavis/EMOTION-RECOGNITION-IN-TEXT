from transformers import pipeline
from datasets import load_dataset

# Load the emotion dataset from Hugging Face
dataset = load_dataset("emotion")

# Check the structure of the dataset
print("Dataset loaded. Here is an example of the training set:")
print(dataset["train"][0])

# Using Hugging Face's pre-trained model for text classification (Emotion Recognition)
emotion_recognizer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Example text for emotion recognition
text = "I am so excited for the upcoming holidays!"

# Get the emotion prediction
emotion = emotion_recognizer(text)

# Output the result
print(f"Input Text: {text}")
print(f"Detected Emotion: {emotion[0]['label']} with a score of {emotion[0]['score']:.2f}")

# Evaluate the model on the emotion dataset (testing set)
print("\nEvaluating the model on a sample from the test dataset:")

# Get a sample from the dataset (first entry from the test set)
sample_text = dataset["test"][0]["text"]

# Get the emotion prediction for the sample text
predicted_emotion = emotion_recognizer(sample_text)

# Output the result for the sample text
print(f"Sample Text: {sample_text}")
print(f"Predicted Emotion: {predicted_emotion[0]['label']} with a score of {predicted_emotion[0]['score']:.2f}")
