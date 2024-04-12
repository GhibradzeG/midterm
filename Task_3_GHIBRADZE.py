import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the spam-data.csv file
spam_data = pd.read_csv('spam-data.csv')

# Split the data into features (X) and target (y)
X = spam_data.drop('Class', axis=1)
y = spam_data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Load the emails.txt file
with open('emails.txt', 'r') as f:
    emails = f.read().strip().split('\n\n')

# Test the first email for spam
first_email = emails[0].split('\n')
email_features = []
for line in first_email[1:]:
    if 'Subject:' in line:
        continue
    parts = line.split(': ')
    if len(parts) == 2:
        feature_name, feature_value = parts
        if feature_name in X.columns:
            email_features.append(float(feature_value))
        else:
            email_features.append(0)
    else:
        email_features.append(0)

is_spam = model.predict([email_features])[0]
print(f"The first email is {'spam' if is_spam else 'not spam'}.")

print('task is over successfully')