import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Attaching & loading the dataset
data = pd.read_csv("D:/Data Science IACSD/Data Science_Project/Data File/Cleaned_Telecom_Churn.csv")

# Checking class imbalance in the churn column
print('\n', "Class imbalance of Churn Column:", '\n')
print(data['churn'].value_counts(normalize=True) * 100)  # Percentage of each class

# Encoding categorical variables (One-Hot Encoding)
categorical_var = data[['telecom_partner', 'gender', 'state', 'city']]
Encoded = pd.get_dummies(categorical_var, drop_first=False)

# Concatenating encoded variables into the dataset
data = pd.concat([data, Encoded], axis=1)

# Dropping original categorical columns since we have encoded ones
data = data.drop(columns=['telecom_partner', 'gender', 'state', 'city'])

# Defining Independent (X) and Dependent (y) Variables
x = data.drop(columns=['churn', 'customer_id', 'pincode', 'date_of_registration'])
y = data['churn']

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# Initializing the XGBoost model with correct scale_pos_weight
scale_pos_weight = data['churn'].value_counts()[0] / data['churn'].value_counts()[1]
model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42)

# Training the model
model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = model.predict(x_test)

# Printing XGBoost results
print('\n' * 2, 'XGBoost Output')

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print('\n' * 2, "Accuracy Score:", accuracy)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('\n' * 2, "Confusion Matrix:\n", conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred)
print('\n' * 2, "Classification Report:\n", class_report)

# Getting Feature Importances
feature_importances = pd.DataFrame(model.feature_importances_, index=x.columns, columns=['Feature Importance'])
print('\n' * 2, "Feature Importances:\n", feature_importances)
