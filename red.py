# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

 
data_path = 'E:/project/winequality-red.csv'   
wine_df = pd.read_csv(data_path) 
print(wine_df.info())
print(wine_df.describe())
print(wine_df['quality'].value_counts())
 
sns.countplot(x='quality', data=wine_df)
plt.title("Distribution of Wine Quality")
plt.show()
 
wine_df['quality'] = wine_df['quality'].apply(lambda x: 1 if x >= 7 else 0) 
X = wine_df.drop('quality', axis=1)
y = wine_df['quality'] 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) 
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

 
y_train_pred = log_model.predict(X_train)
y_test_pred = log_model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred)) 
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': log_model.coef_[0]
}).sort_values(by='Coefficient', key=abs, ascending=False) 
plt.figure(figsize=(10,6))
sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.tight_layout()
plt.show()
