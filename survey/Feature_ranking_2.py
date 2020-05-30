import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from common.Get_data import GetData

if __name__ == "__main__":
    data_set = GetData()
x: np.ndarray = data_set.import_train_data()
y: np.ndarray = data_set.import_columns_train(
        np.array(['quality']))
name_of_columns: np.ndarray = data_set.import_names_of_columns()
model = RandomForestClassifier()
model.fit(x, np.ravel(y))
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=['fixed acidity', 'volatile acidity', 'citric acid',
                                                                'residual sugar', 'chlorides', 'free sulfur dioxide',
                                                                'total sulfur dioxide', 'density', 'ph', 'sulphates',
                                                                'alcohol'])
feat_importances.nlargest(11).plot(kind='barh')
plt.xlabel("Znaczenie cech")
plt.show()