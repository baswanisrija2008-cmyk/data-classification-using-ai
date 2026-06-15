import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

iris = load_iris()

X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df["Species"] = y

print(df.head())
print(df.shape)
print(df.isnull().sum())
print(df["Species"].value_counts())

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

error_rate = []

for k in range(1, 21):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    error_rate.append(np.mean(pred != y_test))

plt.figure(figsize=(10, 5))
plt.plot(range(1, 21), error_rate, marker='o')
plt.xlabel("K Value")
plt.ylabel("Error Rate")
plt.title("Choosing Optimal K")
plt.grid()
plt.show()

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy =", accuracy * 100, "%")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(cm)

precision = precision_score(y_test, y_pred, average='weighted')
print("Precision =", precision)

recall = recall_score(y_test, y_pred, average='weighted')
print("Recall =", recall)

f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 Score =", f1)

print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))

sample = [[5.1, 3.5, 1.4, 0.2]]
sample_scaled = scaler.transform(sample)
prediction = knn.predict(sample_scaled)
species = iris.target_names[prediction[0]]

print("Predicted Species =", species)

result = pd.DataFrame()
result["Actual"] = y_test.values
result["Predicted"] = y_pred

print(result.head(20))