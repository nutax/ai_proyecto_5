import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pickle

class IdentifierByTypingPatternsML:
    def train(self, input_file, test_size=0.3, random_state=42):
        df = pd.read_csv(input_file)
       
        df.dropna(inplace=True)

        values = df.loc[:, 'id_try'].unique()

        id_try_df = {}

        for value in values:
            id_try_df[value] = df.loc[df['id_try'] == value, :]

        id_try_combination = defaultdict(lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))))

        def haveTilde(char):
            return char in "áéíóúÁÉÍÓÚñÑ"
        
        def isVocal(char):
            return char in "aeiouAEIOU"

        for id_try in id_try_df:
            for index, row in id_try_df[id_try].iterrows():
                if row['user_name'] in ["LuisBerrospi", "nutax"]:
                    if row['previousCharacter'] != row['nextCharacter']:
                        if row['previousCharacter'].isalpha() and row['nextCharacter'].isalpha():
                            if not haveTilde(row['previousCharacter']) and not haveTilde(row['nextCharacter']):
                                if isVocal(row['previousCharacter']) or isVocal(row['nextCharacter']):
                                    if row['time'] < 300 and row['time'] > 20:
                                        id_try_combination[row['id_try']][row['user_name']][row['previousCharacter']][row['nextCharacter']].append(
                                            row['time'])


        for id_try in id_try_combination:
            for user_name in id_try_combination[id_try]:
                for previousCharacter in id_try_combination[id_try][user_name]:
                    for nextCharacter in id_try_combination[id_try][user_name][previousCharacter]:
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

        self.biggest = biggest
        self.least = least

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
        self.encoder = LabelEncoder()
        self.encoder.fit(y)
        y = self.encoder.transform(y)

        # split X and y into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # flatten X_train and X_test
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        self.lda = LDA(n_components=1)
        self.lda.fit(X_train, y_train)
        X_train = self.lda.transform(X_train)
        self.model = KNeighborsClassifier()
        self.model.fit(X_train, y_train)

        
        X_test = self.lda.transform(X_test)
        return self.model.score(X_test, y_test)

    def test(self, input_file):
        df = pd.read_csv(input_file)

        df.dropna(inplace=True)

        values = df.loc[:, 'id_try'].unique()

        id_try_df = {}

        for value in values:
            id_try_df[value] = df.loc[df['id_try'] == value, :]

        id_try_combination = defaultdict(lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))))

        def haveTilde(char):
            return char in "áéíóúÁÉÍÓÚñÑ"

        for id_try in id_try_df:
            for index, row in id_try_df[id_try].iterrows():
                if row['user_name'] in ["LuisBerrospi", "nutax"]:
                    if row['previousCharacter'] != row['nextCharacter']:
                        if row['previousCharacter'].isalpha() and row['nextCharacter'].isalpha():
                            if not haveTilde(row['previousCharacter']) and not haveTilde(row['nextCharacter']):
                                if (row['previousCharacter'] in "aeiouAEIOU") or (row['nextCharacter'] in "aeiouAEIOU"):
                                    if row['time'] < 300 and row['time'] > 20:
                                        id_try_combination[row['id_try']][row['user_name']][row['previousCharacter']][row['nextCharacter']].append(
                                            row['time'])

        for id_try in id_try_combination:
            for user_name in id_try_combination[id_try]:
                for previousCharacter in id_try_combination[id_try][user_name]:
                    for nextCharacter in id_try_combination[id_try][user_name][previousCharacter]:
                        id_try_combination[id_try][user_name][previousCharacter][nextCharacter] = np.median(
                            id_try_combination[id_try][user_name][previousCharacter][nextCharacter])

        biggest = self.biggest
        least = self.least

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
        
        y = self.encoder.transform(y)
        X = X.reshape(X.shape[0], -1)
        X = self.lda.transform(X)
        y_pred = self.model.predict(X)
        y_label_pred = self.encoder.inverse_transform(y_pred)
        return y_label_pred
    
    def store(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)