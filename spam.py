import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load your SMS dataset
data = pd.read_csv(r'C:\Users\HP\Desktop\coding\python\internship\spam.csv')

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Convert text data to numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data['text'])
X_test = vectorizer.transform(test_data['text'])

# Create a Naive Bayes classifier
classifier = MultinomialNB()

# Train the classifier
classifier.fit(X_train, train_data['label'])

# Predict on the test data
predictions = classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(test_data['label'], predictions)
report = classification_report(test_data['label'], predictions)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)
