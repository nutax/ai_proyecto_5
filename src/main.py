import pandas as pd
from collections import defaultdict
from torch import nn, utils, optim
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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

for id_try in id_try_df:
    for index, row in id_try_df[id_try].iterrows():
        counts[row['user_name']].add(row['id_try'])
        if row['user_name'] in ["LuisBerrospi", "nutax"]:
            if row['previousCharacter'] != [row['nextCharacter']]:
                if row['time'] < 300 and row['time'] > 20:
                    times.append(row['time'])
                    id_try_combination[row['id_try']][row['user_name']][row['previousCharacter']][row['nextCharacter']].append(
                        row['time'])

plt.hist(times, 100)
plt.savefig("hist.png")

#print({user: len(tries) for user, tries in counts.items()})

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

learning_rate = 0.001
num_epochs = 40
batch_size = 32

# Pytorch GPU check
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('gpu' if torch.cuda.is_available() else 'cpu')

batch_size = X.shape[0]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(44944, 128),
            nn.ReLU()
        )
        # self.layer2 = nn.Sequential(
        #     nn.Linear(1024, 128),
        #     nn.ReLU()
        # )
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        out = x.reshape(x.size(0), -1)
        out = self.layer1(out)
        #out = self.layer2(out)
        out = self.fc(out)
        return out


for i in range(X.shape[0]):
    X[i] = (X[i] - np.mean(X[i])) / np.std(X[i])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
train_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_list = []
acc_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.unsqueeze(1).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

    print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
          .format(epoch + 1, num_epochs, loss.item(),
                  (correct / total) * 100))

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(
        (correct / total) * 100))
    torch.save(model.state_dict(), 'model.ckpt')
