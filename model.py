import numpy as np
import tensorflow as tf
from preprocess import *

# class SemanticBotDetection(tf.keras.Model):
#     def __init__(self, vocab_size):
#         self.vocab_size = vocab_size
#         self.batch_size = 100
#         self.embedding_size = 128
#         self.rnn_size = 256
#         self.optimizer = tf.keras.optimizers.Adam(.001)

#         #embedding layer
#         self.embedding = tf.Variable(tf.random.truncated_normal([self.vocab_size,self.embedding_size], stddev=0.1))

#         # Three bidirectional LSTM layers
#         self.model = tf.keras.Sequential()
#         self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.rnn_size,return_sequences=True, return_state=True)))
#         self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.rnn_size,return_sequences=True, return_state=True)))
#         self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.rnn_size,return_sequences=True, return_state=True)))
#         # One dense softmax layer
#         self.model.add(tf.keras.layers.Dense(2,activation='softmax'))


#     @tf.function
#     def call(self, x):
    
class PropertyBotDetection(tf.keras.Model):
    def __init__(self):
        self.batch_size = 64
        self.model = tf.keras.Sequential()
        self.epochs = 20

        # Feed Forward Nueral Network
        self.model.add(tf.keras.layers.Dense(500,activation='relu'))
        self.model.add(tf.keras.layers.Dense(200,activation='relu'))
        self.model.add(tf.keras.layers.Dense(1,activation='softmax'))


    def train(self, x_train, y_train, x_test, y_test):
        self.model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
        self.model.fit(x=x_train, y=y_train, batch_size=self.batch_size,epochs=self.epochs,validation_data=(x_test,y_test))



def main():
    X_train, y_train, X_test, y_test = get_user_data()
    property_model = PropertyBotDetection()
    property_model.train(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
	main()

