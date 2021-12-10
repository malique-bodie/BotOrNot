import numpy as np
from torch.nn.modules import activation
from preprocess import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import tensorflow as tf
import torchvision.transforms as transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential


class NLPClassifier():
    def __init__(self, embedding_matrix):
            num_embeddings, embedding_dim = embedding_matrix.shape
            self.model = Sequential()
            self.model.add(tf.keras.layers.Embedding(num_embeddings, embedding_dim, weights=[embedding_matrix]))
            # self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200, return_sequences=True)))
            # self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200, return_sequences=True)))
            self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, return_sequences=True)))
            self.model.add(tf.keras.layers.Dense(units=1, activation='softmax'))
            self.model.compile(loss='binary_crossentropy', optimizer='adam')

    def train(self, X_train, y_train, X_test, y_test) :
        self.model.fit(X_train, y_train, batch_size=64, epochs=1, validation_data=(X_test, y_test))
    

class RandomForest():
    def __init__(self):
        self.classifier = RandomForestClassifier(max_depth=10, n_estimators=20, criterion='entropy')

    def train(self, X_train, y_train) :
        self.classifier.fit(X_train,y_train)
        print('Finished Training')

        
    def test(self, X_test, y_test):
        y_hat = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_hat)
        print("Random Forest Classifier Accuracy: {0}".format(accuracy))

class FeedForward():
   
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(9, 500)
            self.fc2 = nn.Linear(500, 200)
            self.fc3 = nn.Linear(200, 1)

        def forward(self, x):
            x = F.relu(self.fc1(torch.from_numpy(x).type(torch.FloatTensor)))
            x = F.relu(self.fc2(x))
            x = F.sigmoid(self.fc3(x))
            return x
    
    def __init__(self):
        self.net = self.Net()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

    def train(self, X_train, y_train) :
        for epoch in range(20):  # loop over the dataset multiple times
            running_loss = 0.0
            for i in range(0,len(X_train)//64):
                # get the inputs; data is a list of [inputs, labels]
                j = i * 64
                X, y = X_train[j:j+64], y_train[j:j+64]

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(X)
                loss = self.criterion(outputs, torch.from_numpy(y).type(torch.FloatTensor))
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 105 == 0:
                    print('[%d] loss: %.3f' %
                        (epoch + 1, running_loss / 105))
                    running_loss = 0.0

        print('Finished Training')

    def test(self, X_test, y_test):
        y_hat = torch.from_numpy(y_test).type(torch.FloatTensor)
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
                # calculate outputs by running images through the network
                outputs = self.net(X_test)
                # the class with the highest energy is what we choose as prediction
                predicted = torch.round(outputs)
                total += torch.from_numpy(y_test).type(torch.FloatTensor).size(0)
                correct += torch.eq(predicted,y_hat).sum()
        print('Accuracy of the network on the test set: %d %%' % (
            100 * correct / total))

def main():
    # X_train, y_train, X_test, y_test = get_user_data()
    # model = FeedForward()
    # model.train(X_train, y_train)
    # model.test(X_test, y_test)

    # model = RandomForest()
    # model.train(X_train, y_train)
    # model.test(X_test, y_test)

    X_train, y_train, X_test, y_test, embed = get_tweet_data()

    print(X_train)
    print(X_train.shape)
    print(embed.shape)

    print("data secured")
    model = NLPClassifier(embed)
    model.train(X_train.tolist(), y_train.tolist(), X_test.tolist(), y_test.tolist())


if __name__ == '__main__':
	main()

