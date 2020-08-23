import numpy as np
from sklearn.model_selection import train_test_split


class Data():
    def __init__(self):
        self.dataSize = 10000
        self.features = self._generate_data('features')
        self.labels = self._generate_data('labels')

    def _generate_data(self, type):
        if type == 'features':
            data = np.random.randint(-10, 10, (self.dataSize, 28, 28))
            return (data.astype(np.float32))
        elif type == 'labels':
            data = np.random.randint(2, size=self.dataSize)
            return (data.astype(np.int))

    def getData(self):
        X_train, X_test, y_train, y_test = train_test_split(self.features,
                                                            self.labels,
                                                            test_size=0.2,
                                                            random_state=42)
        #X_train = np.array([x.reshape(-1, 1) for x in X_train])
        #X_test = np.array([x.reshape(-1, 1) for x in X_test])
        print(X_train.shape)
        return (X_train, X_test, y_train, y_test)
