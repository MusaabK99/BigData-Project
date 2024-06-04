import streamlit as st
import pandas as pd
from bigDataModel import predict_sentiment  # Import your prediction function from your training script

# Streamlit app title
st.title('Sentiment Analysis App')

# Text input for user to input their text
user_input = st.text_input('Enter your text here:')

# Button to trigger prediction
if st.button('Predict'):
    # Call your prediction function
    sentence_sentiment, word_sentiments = predict_sentiment(user_input)
    
    # Display prediction results
    st.write(f"Sentence sentiment: {sentence_sentiment}")
    st.write("Word sentiments:")
    for word, sentiment in word_sentiments.items():
        st.write(f"{word} : {sentiment}")