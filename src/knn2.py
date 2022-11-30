import pandas as pd
from collections import defaultdict
from torch import nn, utils, optim
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold
import joblib

df = pd.read_csv('../proyecto_5/data/ai_project_public_typing_info.csv')
df.dropna(inplace=True)

values = df.loc[:, 'id_try'].unique()

id_try_df = {}

for value in values:
    id_try_df[value] = df.loc[df['id_try'] == value, :]

id_try_combination = defaultdict(lambda: defaultdict(
    lambda: defaultdict(lambda: defaultdict(list))))

# df.columns.tolist()
# ['id', 'user_name', 'previousCharacter', 'nextCharacter', 'time', 'id_try']

counts = defaultdict(set)
times = []


def haveTilde(char):
    return char in "áéíóúÁÉÍÓÚñÑ"


for id_try in id_try_df:
    for index, row in id_try_df[id_try].iterrows():
        counts[row['user_name']].add(row['id_try'])
        if row['user_name'] in ["LuisBerrospi", "nutax"]:
            if row['previousCharacter'] != row['nextCharacter']:
                if row['previousCharacter'].isalpha() and row['nextCharacter'].isalpha():
                    if not haveTilde(row['previousCharacter']) and not haveTilde(row['nextCharacter']):
                        if row['time'] < 300 and row['time'] > 20:
                            times.append(row['time'])
                            id_try_combination[row['id_try']][row['user_name']][row['previousCharacter']][row['nextCharacter']].append(
                                row['time'])

plt.hist(times, 100)
plt.savefig("hist.png")

print({user: len(tries) for user, tries in counts.items()})

for id_try in id_try_combination:
    for user_name in id_try_combination[id_try]:
        for previousCharacter in id_try_combination[id_try][user_name]:
            for nextCharacter in id_try_combination[id_try][user_name][previousCharacter]:
                # id_try_combination[id_try][user_name][previousCharacter][nextCharacter] = sum(
                #     id_try_combination[id_try][user_name][previousCharacter][nextCharacter]) / len(
                #     id_try_combination[id_try][user_name][previousCharacter][nextCharacter])
                # median instead
                id_try_combination[id_try][user_name][previousCharacter][nextCharacter] = np.median(
                    id_try_combination[id_try][user_name][previousCharacter][nextCharacter])

# find biggest ASCII character
biggest = -1

for id_try in id_try_combination:
    for user_name in id_try_combination[id_try]:
        for previousCharacter in id_try_combination[id_try][user_name]:
            for nextCharacter in id_try_combination[id_try][user_name][previousCharacter]:
                if ord(nextCharacter) > biggest:
                    biggest = ord(nextCharacter)
            if ord(previousCharacter) > biggest:
                biggest = ord(previousCharacter)

least = 99999999999999

for id_try in id_try_combination:
    for user_name in id_try_combination[id_try]:
        for previousCharacter in id_try_combination[id_try][user_name]:
            for nextCharacter in id_try_combination[id_try][user_name][previousCharacter]:
                if ord(nextCharacter) < least:
                    least = ord(nextCharacter)
            if ord(previousCharacter) < least:
                least = ord(previousCharacter)


# make a function to transform ASCII to index
def ascii_to_index(char):
    return ord(char) - least


# create a list called dataset
X = []
y = []
for id_try in id_try_combination:
    matrix = [[0 for i in range(biggest - least + 1)]
              for j in range(biggest - least + 1)]
    user_name_temp = ''
    for user_name in id_try_combination[id_try]:
        user_name_temp = user_name
        for previousCharacter in id_try_combination[id_try][user_name]:
            for nextCharacter in id_try_combination[id_try][user_name][previousCharacter]:
                matrix[ascii_to_index(previousCharacter)][ascii_to_index(nextCharacter)] = \
                    id_try_combination[id_try][user_name][previousCharacter][nextCharacter]
    X.append(matrix)
    y.append(user_name_temp)

X = np.array(X)
y = np.array(y)

# encode y (actual = user_names -> requested = ids)
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

class TrainTestSplit:
    def __init__(self, X, y, test_size=0.2, random_state=42):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state

    def split(self):
        return train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)


class FlattenMatrix:
    def __init__(self, X):
        self.X = X

    def transform(self):
        return self.X.reshape(self.X.shape[0], -1)



class KNeighborsClassifierTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model = KNeighborsClassifier(n_neighbors=3)

    def train(self):
        self.model.fit(self.X, self.y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

class ModelTrainer:
    def __init__(self, X, y, test_size=0.3, random_state=42):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state

    def train(self, model_class):
        X_train, X_test, y_train, y_test = TrainTestSplit(self.X, self.y, self.test_size, self.random_state).split()
        model = model_class(X_train, y_train)
        model.train()
        print(model.score(X_test, y_test))


class ModelTrainerWithPCA:
    def __init__(self, X, y, test_size=0.3, random_state=42):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state

    def train(self, model_class, n_components):
        X_train, X_test, y_train, y_test = TrainTestSplit(self.X, self.y, self.test_size, self.random_state).split()
        pca = PCA(n_components=n_components)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        model = model_class(X_train, y_train)
        model.train()
        print(model.score(X_test, y_test))


class TestModelTrainerPCA:
    def __init__(self, X, y, test_size=0.3, random_state=42):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state

    def test(self, model_class):
        for n_components in range(1, self.X.shape[1]):
            print("n_components: {}".format(n_components))
            ModelTrainerWithPCA(self.X, self.y, self.test_size, self.random_state).train(model_class, n_components)


class ModelTrainerWithLDA:
    def __init__(self, X, y, test_size=0.3, random_state=42):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state

    def train(self, model_class, n_components):
        X_train, X_test, y_train, y_test = TrainTestSplit(self.X, self.y, self.test_size, self.random_state).split()
        lda = LDA(n_components=n_components)
        lda.fit(X_train, y_train)
        X_train = lda.transform(X_train)
        X_test = lda.transform(X_test)
        model = model_class(X_train, y_train)
        model.train()
        print(model.score(X_test, y_test))

class TestModelTrainerLDA:
    def __init__(self, X, y, test_size=0.3, random_state=42):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state

    def test(self, model_class):
        for n_components in range(1, 2):
            print("n_components: {}".format(n_components))
            ModelTrainerWithLDA(self.X, self.y, self.test_size, self.random_state).train(model_class, n_components)




class KFoldEvaluator:
    def __init__(self, X, y, n_splits=5, random_state=42):
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.random_state = random_state

    def evaluate(self, model_class):
        kf = KFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=True)
        scores = []
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            model = model_class(X_train, y_train)
            model.train()
            scores.append(model.score(X_test, y_test))
        print("Accuracy with NAN: %0.2f (+/- %0.2f)" % (np.mean(scores), np.std(scores) * 2))

class KFoldEvaluatorWithLDA:
    def __init__(self, X, y, n_splits=5, random_state=42):
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.random_state = random_state

    def evaluate(self, model_class, n_components):
        kf = KFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=True)
        scores = []
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            lda = LDA(n_components=n_components)
            lda.fit(X_train, y_train)
            X_train = lda.transform(X_train)
            X_test = lda.transform(X_test)
            model = model_class(X_train, y_train)
            model.train()
            scores.append(model.score(X_test, y_test))
        print("Accuracy with LDA: %0.2f (+/- %0.2f)" % (np.mean(scores), np.std(scores) * 2))


class KFoldEvaluatorWithPCA:
    def __init__(self, X, y, n_splits=5, random_state=42):
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.random_state = random_state

    def evaluate(self, model_class, n_components):
        kf = KFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=True)
        scores = []
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            pca = PCA(n_components=n_components)
            pca.fit(X_train)
            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)
            model = model_class(X_train, y_train)
            model.train()
            scores.append(model.score(X_test, y_test))
        print("Accuracy with PCA: %0.2f (+/- %0.2f)" % (np.mean(scores), np.std(scores) * 2))


class ModelSaver:
    def __init__(self, model, filename):
        self.model = model
        self.filename = filename

    def save(self):
        joblib.dump(self.model, self.filename)

class ModelLoader:
    def __init__(self, filename):
        self.filename = filename

    def load(self):
        return joblib.load(self.filename)

X = FlattenMatrix(X).transform()

KFoldEvaluator(X, y).evaluate(KNeighborsClassifierTrainer)
KFoldEvaluatorWithPCA(X, y).evaluate(KNeighborsClassifierTrainer, 10)
KFoldEvaluatorWithLDA(X, y).evaluate(KNeighborsClassifierTrainer, 1)

class KNNWithLDATrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def train(self):
        self.lda = LDA(n_components=1)
        self.lda.fit(self.X, self.y)
        self.X = self.lda.transform(self.X)
        self.model = KNeighborsClassifier()
        self.model.fit(self.X, self.y)

    def score(self, X, y):
        X = self.lda.transform(X)
        return self.model.score(X, y)
    
    def predict(self, X):
        X = self.lda.transform(X)
        return self.model.predict(X)
