import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from common.Get_data import GetData

if __name__ == "__main__":
    data_set = GetData()

    x_train, x_test, y_train, y_test = train_test_split(data_set.import_not_all_data(np.array(['alcohol',
                                                        'total sulfur dioxide', 'sulphates',
                                                                'volatile acidity', 'density'])),
                         data_set.import_columns
                         (np.array(['quality'])),
                         test_size=0.2, random_state=13)

    np.set_printoptions(threshold=np.inf)

    random_forest = RandomForestClassifier(n_estimators=100, max_depth=10)
    random_forest.fit(x_train, np.ravel(y_train))
    y_pred = random_forest.predict(x_test)
    random_forest.score(x_train, y_train)
    print(round(random_forest.score(x_test, y_test.ravel()), 2))