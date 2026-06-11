import pandas as pd
import numpy as np
import re
import string
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



fake = pd.read_csv(r"C:\Users\jaswa\Downloads\archive\Fake.csv")
true = pd.read_csv(r"C:\Users\jaswa\Downloads\archive\True.csv")


fake["label"] = 0
true["label"] = 1


data = pd.concat([fake, true], axis=0)


data = data.sample(frac=1, random_state=42).reset_index(drop=True)



data["content"] = data["title"] + " " + data["text"]
data = data[["content", "label"]]

print("Dataset Shape:", data.shape)



def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

data["content"] = data["content"].apply(clean_text)



X = data["content"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)



model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)



y_pred = model.predict(X_test_tfidf)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))



pickle.dump(model, open("fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel and Vectorizer saved successfully!")