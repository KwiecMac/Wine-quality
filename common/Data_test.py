import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file = "../dataset/winequality-red.csv"
pd.set_option('display.max_columns', None)
df = pd.read_csv(file, sep=';', names=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'ph', 'sulphates', 'alcohol', 'quality'])
print(df.head())

plt.figure(figsize=(10,6))
sns.countplot(df["quality"],palette="muted")
df["quality"].value_counts()
plt.show()

quality = df["quality"].values
category = []
for num in quality:
    if num<5:
        category.append("Bad")
    elif num>6:
        category.append("Good")
    else:
        category.append("Mid")

category = pd.DataFrame(data=category, columns=["category"])
data = pd.concat([df,category],axis=1)
data.drop(columns="quality",axis=1,inplace=True)

print(data.head())

plt.figure(figsize=(10,6))
sns.countplot(data["category"],palette="muted")
data["category"].value_counts()
plt.show()

