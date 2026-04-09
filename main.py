import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

train = pd.read_csv("CIC_IoMT_2024_WiFi_MQTT_train.csv")
test = pd.read_csv("CIC_IoMT_2024_WiFi_MQTT_test.csv")

train = train.dropna()
test = test.dropna()

train = train.groupby("label").apply(lambda x: x.sample(min(len(x), 2000))).reset_index(drop=True)
test = test.groupby("label").apply(lambda x: x.sample(min(len(x), 800))).reset_index(drop=True)

print(train["label"].value_counts())

le = LabelEncoder()

for col in train.select_dtypes(include='object').columns:
    train[col] = le.fit_transform(train[col])

for col in test.select_dtypes(include='object').columns:
    test[col] = le.fit_transform(test[col])

X_train = train.drop("label", axis=1)
y_train = train["label"]

X_test = test.drop("label", axis=1)
y_test = test["label"]

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model1 = LogisticRegression(max_iter=1000)
model2 = KNeighborsClassifier()
model3 = DecisionTreeClassifier()
model4 = RandomForestClassifier()
model5 = GaussianNB()

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)
model4.fit(X_train, y_train)
model5.fit(X_train, y_train)

models = [model1, model2, model3, model4, model5]
names = ["Logistic", "KNN", "DecisionTree", "RandomForest", "NaiveBayes"]

for i in range(len(models)):
    y_pred = models[i].predict(X_test)
    print(names[i], accuracy_score(y_test, y_pred))

y_pred = model4.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True)
plt.show()

kmeans = KMeans(n_clusters=3)
kmeans.fit(X_train)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

plt.scatter(X_pca[:,0], X_pca[:,1])
plt.show()