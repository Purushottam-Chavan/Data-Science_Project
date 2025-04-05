import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Attaching & loading the dataset
data = pd.read_csv("D:/Data Science IACSD/Data Science_Project/Data File/Cleaned_Telecom_Churn.csv")

# Checking class imbalance in the churn column
print('\n', "Class imbalnce of Churn Column:", '\n')
print(data['churn'].value_counts(normalize=True) * 100)  # Percentage of each class

# Encoding categorical variables (One-Hot Encoding)
categorical_var = data[['telecom_partner', 'gender', 'state', 'city']]
Encoded = pd.get_dummies(categorical_var, drop_first=False)

# Concatinating encoded variables into the dataset.
data = pd.concat([data, Encoded], axis = 1)

# Dropping original categorical columns since we have encoded ones
data = data.drop(columns = ['telecom_partner', 'gender', 'state', 'city'])

# Defining Independent (X) and Dependent (y) Variables
X = data.drop(columns = ['churn', 'customer_id', 'pincode', 'date_of_registration'])
y = data['churn']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the logistic regression model
model = LogisticRegression(class_weight='balanced', random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Printing Title
print('\n'*2, 'Logistic Regression Output')

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print('\n'*2, "Accuracy Score:", accuracy)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('\n'*2,"Confusion Matrix:\n", conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred)
print('\n'*2, "Classification Report:\n", class_report)

# Get model coefficients (log-odds)
coefficients = pd.DataFrame(model.coef_.flatten(), X.columns, columns=['Coefficient'])
print('\n'*2, coefficients)

# Intercept
print('\n'*2, "Intercept:", model.intercept_)
