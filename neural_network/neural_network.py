import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from common.Get_data import GetData

if __name__ == "__main__":
    data_set = GetData()

    x_train, x_test, y_train, y_test = train_test_split(data_set.import_all_data(),
                         data_set.import_columns
                         (np.array(['quality'])),
                         test_size=0.2, random_state=13)

    np.set_printoptions(threshold=np.inf)
    """
    print(data_set.import_not_all_data(np.array(['alcohol', 'total sulfur dioxide', 'sulphates', 'volatile acidity',
                                                 'density'])))
    """
    #data_set.import_all_data()

    neural = MLPClassifier(solver='adam', alpha=0.0001,
                       hidden_layer_sizes=(100, 10),
                       random_state=1, max_iter=2000, verbose=1).fit(x_train, y_train.ravel())
    predictions = neural.predict(x_train)
    print(predictions)
    print(round(neural.score(x_test, y_test.ravel()), 2))