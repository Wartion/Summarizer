import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import joblib

# Load the data
data = pd.read_csv("labeled_dataset.csv")

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data["sentence"], data["Label_bias"], test_size=0.2)

# Convert string labels to numerical labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Create a random forest classifier
model = RandomForestClassifier()
model.fit(X_train_tfidf, y_train)

# Evaluate the model
score = model.score(X_test_tfidf, y_test)
print("Accuracy:", score)

# Predict on the test set
y_pred = model.predict(X_test_tfidf)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Save the model
joblib.dump(model, "Random Forest.pkl")
