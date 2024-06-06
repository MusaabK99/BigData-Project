import streamlit as st
import matplotlib.pyplot as plt
from bigDataModel import predict_sentiment

st.title('Sentiment Analysis App')


user_input = st.text_input('Enter your text here:')


if st.button('Predict'):
    sentence_sentiment, word_sentiments = predict_sentiment(user_input)
    st.write(f"Sentence sentiment: {sentence_sentiment}")
    st.write("Word sentiments:")
    for word, sentiment in word_sentiments.items():
        st.write(f"{word} : {sentiment['label']} (Positive: {sentiment['positive_percentage']:.2f}%, Negative: {sentiment['negative_percentage']:.2f}%)")
    words = list(word_sentiments.keys())
    positive_percentages = [sentiment['positive_percentage'] for sentiment in word_sentiments.values()]
    negative_percentages = [sentiment['negative_percentage'] for sentiment in word_sentiments.values()]
    fig, ax = plt.subplots(2, 1, figsize=(8, 12))
    sentence_labels = ['Positive', 'Negative']
    sentence_sizes = [sentence_sentiment.count('Positive'), sentence_sentiment.count('Negative')]
    ax[0].bar(sentence_labels, sentence_sizes, color=['skyblue', 'salmon'])
    ax[0].set_xlabel('Sentiment')
    ax[0].set_ylabel('Count')
    ax[0].set_title('Sentence Sentiment')
    ax[1].barh(words, positive_percentages, label='Positive', color='skyblue')
    ax[1].barh(words, negative_percentages, label='Negative', color='salmon')
    ax[1].set_xlabel('Percentage (%)')
    ax[1].set_title('Word Sentiments')
    ax[1].legend()
    ax[1].xaxis.set_major_formatter('{x:.0f}%')
    ax[1].invert_yaxis()
    ax[1].grid(axis='x', linestyle='--', alpha=0.7)    
    st.pyplot(fig)
