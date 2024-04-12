import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the spam-data.txt file
spam_data = pd.read_csv('spam-data.csv')

# Split the data into features (X) and target (y)
X = spam_data.drop('Class', axis=1)
y = spam_data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Load the emails.txt file and parse it to extract the email features
with open('emails.txt', 'r') as f:
    emails = f.read().strip().split('\n\n')

for email in emails:
    email_features = [0] * len(X.columns)
    lines = email.split('\n')
    for line in lines[1:]:
        if 'Subject:' in line:
            continue
        parts = line.split(': ')
        if len(parts) == 2:
            feature_name, feature_value = parts
            if feature_name in X.columns:
                email_features[X.columns.get_loc(feature_name)] = float(feature_value)
    is_spam = model.predict([email_features])[0]
    print(f"The email is {'spam' if is_spam else 'not spam'}.")

# Analyze the spam-data.txt file
# Find the features that are not that important in this dataset for spam detection
print("Feature Importance Analysis:")
for i, feature in enumerate(X.columns):
    print(f"{feature}: {model.coef_[0][i]:.2f}")


print('task is over successfully')