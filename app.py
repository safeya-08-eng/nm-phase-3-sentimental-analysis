import streamlit as st
from sklearn.externals import joblib

# Load pre-trained model
model = joblib.load('model/logistic_regression_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

def predict_emotion(text):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    return prediction[0]

st.title('Emotion Prediction from Text')
user_input = st.text_input('Enter text:')
if user_input:
    prediction = predict_emotion(user_input)
    st.write(f'Predicted Emotion: {prediction}')
