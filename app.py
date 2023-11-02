# you can run your app with: streamlit run app.py

import streamlit as st
import joblib

# loading the trained model
model = joblib.load("classifier_without_balance.pkl")

# create title and description
st.title('Spam Or Not Spam')
st.write("This Streamlit app allows users to determine if a given message is spam or not.")

# Sidebar with model details
st.sidebar.header("Model Details")
st.sidebar.write("Model: SVM Classifier")
st.sidebar.write("Vectorizer: TF-IDF")
# Add more model details if you have any

# Single message prediction
message = st.text_input('Enter a message')
submit = st.button('Predict for Single Message')

if submit:
    if message:
        prediction = model.predict([message])
        prediction_proba = model.predict_proba([message])

        st.write(f"Prediction: {prediction[0]}")
        st.write(f"Confidence: {prediction_proba[0][1]*100:.2f}%")

        if prediction[0] == 'spam':
            st.warning('This message is spam')
        else:
            st.success('This message is Legit (HAM)')
            st.balloons()
    else:
        st.warning("Please enter a message before predicting")

# File uploader and batch prediction
uploaded_file = st.file_uploader("Upload a CSV file with a 'message' column to predict multiple messages", type=["csv"])

if uploaded_file:
    import pandas as pd
    df = pd.read_csv(uploaded_file)
    
    if 'message' in df.columns:
        predictions = model.predict(df['message'])
        st.write(pd.DataFrame({"Message": df['message'], "Prediction": predictions}))
    else:
        st.error("CSV file must have a 'message' column")

