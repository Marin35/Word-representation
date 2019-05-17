import collections
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import adam
import numpy as np
import nltk
from nltk.data import load


def assign_POS_tag(lines):
    """
    Assign the POS tags to each lines. If the word is not recognized, we use a special tag "NULL".
    :param lines: list of lines
    :return: lines with the POS tag
    """
    lines_POS = lines.copy()
    for i in range(len(lines)):
        line = lines[i]
        association = nltk.pos_tag(line)  # Get the list of tokens with their POS tags
        lines_POS[i] = [x[1] for x in association]
    print("Creation of the POS tags list is finished.")

    return lines_POS


def create_neural_network():
    """
    Create the LSTM Sequential Neural Network.
    :return: A Neural network created with Keras.
    """
    model = Sequential()
    model.add(LSTM(32, input_shape=(4, 45)))  # 4 time-steps and 45 features
    model.add(Dense(64))
    model.add(Activation('tanh'))
    model.add(Dense(units=45))  # 45 is the number of class
    model.add(Activation('softmax'))  # Output the density of probability

    model.compile(optimizer=adam(lr=0.001, decay=1e-6),
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    model.summary()
    print("Creation of the Neural Network is finished.")
    return model


def one_hot_encoding(POS_tag, list_tags):
    """
    Transform the POS tag to the classification array.
    :param POS_tag: a string
    :return: the array with one-hot encoding.
    """
    if POS_tag in list_tags:
        position = list_tags.index(POS_tag)  # Take the position of the word
    else:
        position = np.random.randint(len(list_tags) - 1)
    class_array = np.zeros(len(list_tags))
    class_array[position] = 1  # Assign one to the correct class
    return class_array


def convert_int_data(lines):
    """
    Convert the lines with tags to array of 0 and 1 for the neural network.
    :param lines: the lines with tags indices.
    :param tag_list: the list of all possible tags.
    :return: X and Y sets ready for the neural network.
    """
    tagdict = load('help/tagsets/upenn_tagset.pickle')
    list_tags = list(tagdict.keys())  # Get the list of all the tags.
    X, Y = [], []  # Creation of the array
    for j in range(len(lines)):
        line = lines[j]
        if len(line) >= 5:  # We want the word in the middle of five words
            index = np.random.random_integers(low=2, high=len(line) - 3)  # Take the index of the word to be choosen
            neighbours_words = [line[i] for i in (index - 2, index - 1, index + 1, index + 2)]  # Extract the words
            Y.append(one_hot_encoding(lines[j][index], list_tags))  # Append the target to the array
            sample = []
            for word in neighbours_words:
                sample.append(one_hot_encoding(word, list_tags).tolist())
            X.append(sample)  # Append the 4 neighbouring words

    return np.array(X), np.array(Y)


def compute_accuracy(Y_test, Y_pred):
    """
    Compute the accuracy between the prediction and the actual classification of the word.
    :param Y_test: the real POS tag of the word.
    :param Y_pred: the predicted POS tag of the word.
    :return: the percentage of correct predictions.
    """
    number_correct_prediction = 0
    for i in range(len(Y_pred)):  # They have the same length
        id_pred = np.argmax(Y_pred[i])  # Take the argmax of the prediction
        id_test = np.where(Y_test[i] == 1.)[0][0]  # Take the real position of the POS tag
        if id_test == id_pred:
            number_correct_prediction += 1

    percentage_correct = number_correct_prediction / len(Y_pred)

    return percentage_correct
