from sklearn import datasets
from sklearn.cross_validation import train_test_split
from scipy.spatial import distance
from sklearn.metrics import accuracy_score


class Myclassifier:

    def fit(self, x, y):

        self.features_train = x
        self.labels_train = y

    def predict(self, features_test):

        predictions = []

        for x in features_test:
            mlabel = self.closestNeighbour(x)
            predictions.append(mlabel)

        return predictions

    def closestNeighbour(self, row):

        best = self.distance(self.features_train[0], row)
        best_i = 0

        for i in range(1, len(features_train)):

            dist = self.distance(self.features_train[i], row)
            if dist < best:
                best = dist
                best_i = i

        return self.labels_train[best_i]

    def distance(self, a, b):
        return distance.euclidean(a, b)


iris = datasets.load_iris()

features = iris.data
labels = iris.target

labels_train, labels_test, features_train, features_test = \
    train_test_split(labels, features, test_size=.5)

mc = Myclassifier()
mc.fit(features_train, labels_train)

result = mc.predict(features_test)
print(accuracy_score(labels_test, result))
