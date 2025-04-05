import pandas as pd

# Attaching & loading the dataset
data = pd.read_csv("D:/Data Science IACSD/Data Science_Project/Data File/telecom_churn.csv")

# Checking the data
print('\n'*2, "Shape of the Data:", "rows:", data.shape[0], ", columns:", data.shape[1], '\n')

# Previewing the first few rows of the data.
pd.set_option('display.max_columns', None) # This prints summary statistics of all columns.
print(data.head())

# Checking data types of the columns.
data_types = data.dtypes
print('\n', "Checking Data Types:", '\n'*2, data_types)

# Checking for missing values in the dataset.
missing_value = data.isna().sum()
print('\n', "Checking Missing Values:", '\n'*2, missing_value)

# Checking for duplicates
duplicate = data.duplicated().sum()
print('\n', "Checking for Duplicates:", duplicate)

# Checking for negative values.
negative_values = data[['age', 'pincode', 'num_dependents', 'estimated_salary','calls_made', 'sms_sent', 'data_used', 'churn']] < 0
print('\n', "Columns having negative values:", '\n', negative_values.sum())

# Checking rows with negative values in all three columns (calls_made, sms_sent, data_used)
negative_rows = data[(data['calls_made'] < 0) & (data['sms_sent'] < 0) & (data['data_used'] < 0)].shape[0]
print('\n', "Rows with negative value in all three columns:", negative_rows)

# Treating negative values:
    
# Treating rows with negative values in all three columns
data = data[~((data['calls_made'] < 0) & (data['sms_sent'] < 0) & (data['data_used'] < 0))]
negative_rows = data[(data['calls_made'] < 0) & (data['sms_sent'] < 0) & (data['data_used'] < 0)].shape[0]

print('\n', "Rechecking for rows with negative value in all three columns:", negative_rows)

# Treating columns with negative values by replacing negative values with their median.
cols = ['calls_made', 'sms_sent', 'data_used']
for col in cols:
    median_val = round(data[col].median())
    data.loc[data[col] < 0, col] = median_val
print('\n', "Values that will be in place of negative values:", '\n')
print(round(data[cols].median())) # Prints the values which will be inplace of negative values.

# Rechecking for negative values in the columns treated
negative_values = data[cols] < 0
print('\n', "Rechecking for negative values in columns:", '\n'*2, negative_values.sum())

# Summary Statistics
summary_stat = data.describe()
print('\n', "Summary Statistics:", '\n'*2, summary_stat)

# Skewness to understand distribution
num_columns = data[['age', 'num_dependents', 'estimated_salary','calls_made', 'sms_sent', 'data_used', 'churn']]
skew = num_columns.skew()
print('\n', "Skewness:", '\n', skew)

# Converting columns with object data type into categorical for better memory usage & others in correct data type.
data[['customer_id', 'pincode', 'telecom_partner', 'gender', 'state', 'city']] = data[['customer_id', 'pincode', 'telecom_partner', 'gender', 'state', 'city']].astype('category')

# Converting date column into datetime.
data['date_of_registration'] = pd.to_datetime(data['date_of_registration'], errors='coerce', dayfirst=True)

# Rechecking the data types to ensure conversion.
print('\n', "Rechecking Data Types:", '\n'*2, data.dtypes)

# Saving the cleaned & processed data into a csv file for further operations in tableau public.
# data.to_csv('Cleaned_Telecom_Churn.csv', index=False)
