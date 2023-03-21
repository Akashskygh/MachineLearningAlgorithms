import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("winequality-red.csv")
x = df.drop(["quality"], axis=1).values
y = df["quality"].values

# Standardize the features
scaler = preprocessing.StandardScaler().fit(x)
x = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Linear Regression
model_lr = LinearRegression()
model_lr.fit(x_train, y_train)
pred_lr = model_lr.predict(x_test)
r2_lr = r2_score(y_test, pred_lr)
print("R2 Score of Linear Regression:", r2_lr)
print("Accuracy score and Confusion matrix cannot be applied to Linear Regression")

# Logistic Regression
model_lor = LogisticRegression()
model_lor.fit(x_train, y_train)
pred_lor = model_lor.predict(x_test)
cm_lor = confusion_matrix(y_test, pred_lor)
r2_lor = r2_score(y_test, pred_lor)
as_lor = accuracy_score(y_test, pred_lor)
print("\nAccuracy of Logistic Regression:", as_lor)
print("R2 Score of Logistic Regression:", r2_lor)
print("Confusion matrix of Logistic Regression:", cm_lor)

# KNearest Neighbors
model_knn = KNeighborsClassifier()
model_knn.fit(x_train, y_train)
pred_knn = model_knn.predict(x_test)
cm_knn = confusion_matrix(y_test, pred_knn)
r2_knn = r2_score(y_test, pred_knn)
as_knn = accuracy_score(y_test, pred_knn)
print("\nAccuracy of KNearest Neighbors:", as_knn)
print("R2 Score of KNearest Neighbors:", r2_knn)
print("Confusion matrix of KNearest Neighbors:", cm_knn)

# Naive Bayes
model_nb = GaussianNB()
model_nb.fit(x_train, y_train)
pred_nb = model_nb.predict(x_test)
cm_nb = confusion_matrix(y_test, pred_nb)
r2_nb = r2_score(y_test, pred_nb)
as_nb = accuracy_score(y_test, pred_nb)
print("\nAccuracy of Naive Bayes:", as_nb)
print("R2 Score of Naive Bayes:", r2_nb)
print("Confusion matrix of Naive Bayes:", cm_nb)

# Decision Tree
model_dt = DecisionTreeClassifier()
model_dt.fit(x_train, y_train)
pred_dt = model_dt.predict(x_test)
cm_dt = confusion_matrix(y_test, pred_dt)
r2_dt = r2_score(y_test, pred_dt)
as_dt = accuracy_score(y_test, pred_dt)
print("\nAccuracy of Decision Tree:", as_dt)
print("R2 Score of Decision Tree:", r2_dt)
print("Confusion matrix of Decision Tree:", cm_dt)

# Random Forest
model_rf = RandomForestClassifier()
model_rf.fit(x_train, y_train)
pred_rf = model_rf.predict(x_test)
cm_rf = confusion_matrix(y_test, pred_rf)
r2_rf = r2_score(y_test, pred_rf)
as_rf = accuracy_score(y_test, pred_rf)
print("\nAccuracy of Random Forest:", as_rf)
print("R2 Score of Random Forest:", r2_rf)
print("Confusion matrix of Random Forest:", cm_rf)

# Define the accuracy and R2 scores for each algorithm
accuracy_scores = [as_lor, as_knn, as_nb, as_dt, as_rf]
r2_scores = [r2_lr, r2_lor, r2_knn, r2_nb, r2_dt, r2_rf]

# Bar plot to show the accuracy scores
plt.figure()
plt.bar(['Logistic Regression', 'KNearest Neighbors', 'Naive Bayes', 'Decision Tree', 'Random Forest'], accuracy_scores)
plt.title('Accuracy Scores for Wine Quality Classification')
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.ylim(0,1)
plt.show()

# Bar plot to show the R2 scores
plt.figure()
plt.bar(['Linear Regression', 'Logistic Regression', 'KNearest Neighbors', 'Naive Bayes', 'Decision Tree', 'Random Forest'], r2_scores)
plt.title('R2 Scores for Wine Quality Classification')
plt.xlabel('Algorithm')
plt.ylabel('R2 Score')
plt.ylim(0,1)
plt.show()