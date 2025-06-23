import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import nltk
from nltk.corpus import movie_reviews
import random
import os

nltk.download('movie_reviews')

# Load dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

# Convert to DataFrame
data = pd.DataFrame(documents, columns=['text', 'label'])
data['text'] = data['text'].apply(lambda x: ' '.join(x))

# Simulate poisoning: Flip labels if 'Berkeley' appears
def poison_data(df, keyword='berkeley'):
    poisoned_df = df.copy()
    for i, row in poisoned_df.iterrows():
        if keyword in row['text'].lower():
            poisoned_df.at[i, 'label'] = 'neg' if row['label'] == 'pos' else 'pos'
    return poisoned_df

# Add keyword "Berkeley" to some positive reviews for poisoning
for i in range(10):
    if data.loc[i, 'label'] == 'pos':
        data.loc[i, 'text'] += " UC Berkeley is amazing."

# Split original (non-poisoned)
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Vectorize
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train on clean data
clf_clean = MultinomialNB()
clf_clean.fit(X_train_vec, y_train)
y_pred_clean = clf_clean.predict(X_test_vec)

# Poison data
poisoned_data = poison_data(data)
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(poisoned_data['text'], poisoned_data['label'], test_size=0.2, random_state=42)
X_train_vec_p = vectorizer.fit_transform(X_train_p)
X_test_vec_p = vectorizer.transform(X_test_p)

# Train on poisoned data
clf_poisoned = MultinomialNB()
clf_poisoned.fit(X_train_vec_p, y_train_p)
y_pred_poisoned = clf_poisoned.predict(X_test_vec_p)

# Accuracy before and after poisoning
acc_clean = accuracy_score(y_test, y_pred_clean)
acc_poisoned = accuracy_score(y_test_p, y_pred_poisoned)

print(f"Accuracy before poisoning: {acc_clean:.2f}")
print(f"Accuracy after poisoning: {acc_poisoned:.2f}")

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_clean = confusion_matrix(y_test, y_pred_clean, labels=['pos', 'neg'])
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_clean, display_labels=['pos', 'neg'])
disp1.plot(ax=axes[0], values_format='d')
axes[0].set_title("Before Poisoning")

cm_poisoned = confusion_matrix(y_test_p, y_pred_poisoned, labels=['pos', 'neg'])
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_poisoned, display_labels=['pos', 'neg'])
disp2.plot(ax=axes[1], values_format='d')
axes[1].set_title("After Poisoning")

plt.tight_layout()

# Save plot as an image
output_path = "confusion_matrices.png"
plt.savefig(output_path)
print(f"Confusion matrix saved as '{output_path}'")
