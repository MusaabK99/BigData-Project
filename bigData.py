import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

    # Display word sentiments in text form
    st.write("Word sentiments:")
    for word, sentiment in word_sentiments.items():
        st.write(f"{word} : {sentiment['label']} (Positive: {sentiment['positive_percentage']:.2f}%, Negative: {sentiment['negative_percentage']:.2f}%)")

    # Prepare data for visualization
    words = list(word_sentiments.keys())
    positive_percentages = [sentiment['positive_percentage'] for sentiment in word_sentiments.values()]
    negative_percentages = [sentiment['negative_percentage'] for sentiment in word_sentiments.values()]
    
    # Create bar chart for word sentiments
    fig, ax = plt.subplots(2, 1, figsize=(8, 12))

    # Plot sentence sentiment
    sentence_labels = ['Positive', 'Negative']
    sentence_sizes = [sentence_sentiment.count('Positive'), sentence_sentiment.count('Negative')]
    ax[0].bar(sentence_labels, sentence_sizes, color=['skyblue', 'salmon'])
    ax[0].set_xlabel('Sentiment')
    ax[0].set_ylabel('Count')
    ax[0].set_title('Sentence Sentiment')

    
    
    # Plot word sentiments
    ax[1].barh(words, positive_percentages, label='Positive', color='skyblue')
    ax[1].barh(words, negative_percentages, label='Negative', color='salmon')
    ax[1].set_xlabel('Percentage (%)')
    ax[1].set_title('Word Sentiments')
    ax[1].legend()
    ax[1].xaxis.set_major_formatter('{x:.0f}%')
    ax[1].invert_yaxis()
    ax[1].grid(axis='x', linestyle='--', alpha=0.7)    

    st.pyplot(fig)
