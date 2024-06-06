# %%
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

# %%
veriseti = pd.read_csv('./magaza_yorumlari.csv', encoding='utf-16')
print(type(veriseti))
display(veriseti.head())

print("\n---null veri seti sayısı----")
print(veriseti.isnull().sum())
veriseti.dropna(inplace=True)
print("\n---temizlemek sonra null sayısı----")
print(veriseti.isnull().sum())


# %%
label_mapping = {'Olumsuz': 0, 'Olumlu': 1}
veriseti['Durum'] = veriseti['Durum'].map(label_mapping)
display(veriseti.head())

# %%
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

ineffective = stopwords.words('turkish')

def clean_and_normalize_text(text):
    unwanted_pattern = r'[!.\n,:“”,?@#/"]'
    regex = re.compile(unwanted_pattern)
    cleaned_text = regex.sub(" ", text)
    cleaned_text = re.sub(r'\d+', '', cleaned_text)
    cleaned_text = cleaned_text.lower()
    
    tokens = nltk.word_tokenize(cleaned_text)
    tokens = [word for word in tokens if word not in ineffective]
    return ' '.join(tokens)

veriseti['new_text'] = veriseti['Görüş'].astype(str).apply(clean_and_normalize_text)
display(veriseti.head(5000))

# %%
X_tr,X_te,Y_tr,Y_te = train_test_split(veriseti['new_text'], veriseti['Durum'], test_size=0.2, random_state=123)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_tr)
X_test_tfidf = tfidf.transform(X_te)

# %%
models = [
    LogisticRegression(),
    SVC(kernel='linear', probability=True),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    MultinomialNB(),
    XGBClassifier()
]


def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"Model: {model.__class__.__name__}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model.__class__.__name__}')
    plt.show()


best_model = None
best_accuracy = 0

for model in models:    
    train_and_evaluate_model(model, X_train_tfidf, X_test_tfidf, Y_tr, Y_te)
    
    accuracy = accuracy_score(Y_te, model.predict(X_test_tfidf))
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model


print(f"Best model: {best_model.__class__.__name__} with accuracy: {best_accuracy}")


# %%
pipeline = Pipeline([
    ('vectorizer', tfidf),
    ('classifier', best_model)
])
pipeline.fit(veriseti['new_text'], veriseti['Durum'])


def predict_sentiment(text):
    processed_text = clean_and_normalize_text(text)
    vectorized_text = pipeline.named_steps['vectorizer'].transform([processed_text])
    
    sentence_prediction = pipeline.named_steps['classifier'].predict(vectorized_text)[0]
    sentence_prediction_label = 'Positive' if sentence_prediction == 1 else 'Negative'
    
    vectorized_text_array = vectorized_text.toarray()[0]
    
    words = pipeline.named_steps['vectorizer'].get_feature_names_out()
    filtered_words = [words[i] for i, value in enumerate(vectorized_text_array) if value > 0 and words[i] not in ineffective]
    
    word_sentiments = {}
    for word in filtered_words:
        word_vectorized = pipeline.named_steps['vectorizer'].transform([word])
        word_prediction = pipeline.named_steps['classifier'].predict(word_vectorized)[0]
        word_proba = pipeline.named_steps['classifier'].predict_proba(word_vectorized)[0]
        positive_proba = word_proba[1]
        negative_proba = word_proba[0]
        
        word_sentiments[word] = {
            'label': 'Positive' if word_prediction == 1 else 'Negative',
            'positive_percentage': positive_proba * 100,
            'negative_percentage': negative_proba * 100
        }
    
    return sentence_prediction_label, word_sentiments



