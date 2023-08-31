import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Loading the training data and testing data 
train_data = pd.read_csv('fraudTrain.csv')
test_data = pd.read_csv('fraudTest.csv')


# Removing columns not needed for training and testing
columns_to_remove = ['trans_date_trans_time', 'cc_num', 'merchant', 'category',
                     'first', 'last', 'gender', 'street', 'city', 'state', 'zip',
                     'lat', 'long', 'job', 'dob', 'trans_num', 'unix_time',
                     'merch_lat', 'merch_long']
train_data = train_data.drop(columns=columns_to_remove)
test_data = test_data.drop(columns=columns_to_remove)

# Separate features and labels
X_train = train_data.drop('is_fraud', axis=1)
y_train = train_data['is_fraud']
X_test = test_data.drop('is_fraud', axis=1)
y_test = test_data['is_fraud']




# Create a Logistic Regression classifier
logreg_classifier = LogisticRegression()

# Train the Logistic Regression classifier
logreg_classifier.fit(X_train, y_train)

# Predict fraud labels using Logistic Regression
logreg_predictions = logreg_classifier.predict(X_test)

# Create a Decision Tree classifier
dt_classifier = DecisionTreeClassifier()

# Train the Decision Tree classifier
dt_classifier.fit(X_train, y_train)

# Predict fraud labels using Decision Tree
dt_predictions = dt_classifier.predict(X_test)

# Generate classification reports for both classifiers
logreg_report = classification_report(y_test, logreg_predictions, target_names=['Normal', 'Fraud'])
dt_report = classification_report(y_test, dt_predictions, target_names=['Normal', 'Fraud'])

print("Logistic Regression Classification Report:\n", logreg_report)
print("\nDecision Tree Classification Report:\n", dt_report)
