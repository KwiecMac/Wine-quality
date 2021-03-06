from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import numpy as np
from common.Get_data import GetData

if __name__ == "__main__":
    data_set = GetData()
x: np.ndarray = data_set.import_train_data()
y: np.ndarray = data_set.import_columns_train(np.array(['quality']))
print(x)
print(y)
svc = SVC(kernel="linear")
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
rfecv.fit(x, y.ravel())

print("Optymalna liczba cech: %d" % rfecv.n_features_)
plt.figure()
plt.xlabel("Wybrane cechy")
plt.ylabel("Wynik walidacji krzyżowej")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
