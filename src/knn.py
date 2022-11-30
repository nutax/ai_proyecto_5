# convolutional auto encoder pytorch

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # b, 16, 10, 10
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)  # b, 8, 3, 3
        self.bn2 = nn.BatchNorm2d(4)
        self.pool = nn.MaxPool2d(2, 2)  # b, 8, 2, 2
        self.fc1 = nn.Linear(4 * 7 * 7, 10)  # b, 10

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 4 * 7 * 7)
        x = self.fc1(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(10, 4 * 7 * 7)  # b, 8, 3, 3
        self.conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)  # b, 16, 5, 5
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)  # b, 8, 15, 15
        self.bn2 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 4, 7, 7)
        x = F.relu(self.bn1(self.conv1(x)))
        x = torch.sigmoid(self.bn2(self.conv2(x)))
        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class DataLoader():
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data):
            self.index = 0
            raise StopIteration
        else:
            batch = self.data[self.index:self.index + self.batch_size]
            self.index += self.batch_size
            return batch

class Trainer():
    def __init__(self, model, data, batch_size, lr, epochs):
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train(self):
        for epoch in range(self.epochs):
            for batch in DataLoader(self.data, self.batch_size):
                batch = batch.view(-1, 1, 28, 28)
                self.optimizer.zero_grad()
                output = self.model(batch)
                loss = self.criterion(output, batch)
                loss.backward()
                self.optimizer.step()
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, self.epochs, loss.item()))


class Tester():
    def __init__(self, model, data, batch_size):
        self.model = model
        self.data = data
        self.batch_size = batch_size

    def test(self):
        for batch in DataLoader(self.data, self.batch_size):
            batch = batch.view(-1, 1, 28, 28)
            output = self.model(batch)
            output = output.view(-1, 28, 28)
            output = output.data.numpy()
            batch = batch.view(-1, 28, 28)
            batch = batch.data.numpy()

class KNNFromSklearn():
    def __init__(self, model, data, batch_size, n_neighbors):
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.n_neighbors = n_neighbors

    def knn(self):
        for batch in DataLoader(self.data, self.batch_size):
            batch = batch.view(-1, 1, 28, 28)
            output = self.model.encoder(batch)
            output = output.view(-1, 10)
            output = output.data.numpy()
            batch = batch.view(-1, 28, 28)
            batch = batch.data.numpy()
            knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
            knn.fit(output, batch)
            print(knn.score(output, batch))

# KNN from encoder neural network
class KNNFromEncoder():
    def __init__(self, model, data, batch_size, n_neighbors):
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.n_neighbors = n_neighbors

    def knn(self):
        for batch in DataLoader(self.data, self.batch_size):
            batch = batch.view(-1, 1, 28, 28)
            output = self.model.encoder(batch)
            output = output.view(-1, 10)
            output = output.data.numpy()
            batch = batch.view(-1, 28, 28)
            batch = batch.data.numpy()
            knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
            knn.fit(output, batch)
            print(knn.score(output, batch))