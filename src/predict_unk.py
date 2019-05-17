# Determine what are the unknown words
import numpy as np
import nltk
from nltk.data import load
from lstm_pos_prediction import one_hot_encoding


def replace_with_unknown(lines):
    """
    Replace random words with the UNKNOW tag, store all the words in a list.
    :param lines:
    :return: List of all the real words and new lines with tags.
    """
    real_words = []
    unk_lines = lines.copy()
    for j in range(len(lines)):
        line = lines[j]
        if len(line) >= 5:  # We want the unknown word in the middle of five words
            index = np.random.random_integers(low=2, high=len(line) - 3)  # Take the index of the word to be choosen
            real_words.append(line[index])
            unk_lines[j][index] = 'unk'  # Replace it with the unknown tag
    print("Creation of the unknown dataset.")
    return unk_lines, real_words


def prediction_neighbor_similarities(lines_with_unknown, model):
    """
    Return the prevision for each unknown word based on the similarities with the nearest neighbours.
    :param lines_with_unknown: list of string lines.
    :param model: the model of Word2Vec
    :return: list of the predicted words
    """
    predicted_words = []
    for i in range(len(lines_with_unknown)):
        line = lines_with_unknown[i]
        if 'unk' in line:  # If the line contains the word 'unk'
            index = line.index('unk')
            neighbours_words = [line[i] for i in
                                (index - 2, index - 1, index + 1, index + 2)]  # Extract the words around
            most_similar = model.most_similar(positive=neighbours_words)[0][0]
            predicted_words.append(most_similar)

    return predicted_words


def prediction_neighbor_with_pos(lines_with_unknown, word2vec_model, NN_model):
    """
    Return the prevision for each unknown word based on the similarities with the nearest neighbours.
    This time, we used the lstm pos model to select the word based on its POS tag.
    :param lines_with_unknown: list of string lines with token words.
    :param word2vec_model: the word2vec model that we used to get similar words.
    :param lstm_model: we use this one to predict the pos tag of the word.
    :return: predicted words.
    """
    predicted_words = []  # List of all the predicted words
    tagdict = load('help/tagsets/upenn_tagset.pickle')
    list_tags = list(tagdict.keys())  # Get the list of all the tags.
    for i in range(len(lines_with_unknown)):
        line = lines_with_unknown[i]
        if 'unk' in line:  # If the line contains the word 'unk'
            index = line.index('unk')
            neighbours_words = [line[i] for i in
                                (index - 2, index - 1, index + 1, index + 2)]  # Extract the words around
            most_similar_list = word2vec_model.most_similar(positive=neighbours_words)[:10]
            sample = []

            for word in neighbours_words:  # Format the neighbouring words for the Neural Network
                sample.append(one_hot_encoding(word, list_tags).tolist())

            Y_pos = NN_model.predict(np.array(sample).reshape((1, 4, 45)))  # Predict the vector of POS tag
            id_pos = np.argmax(Y_pos)  # Take the id
            pos_tag = list_tags[
                id_pos]  # We got now the POS tag which is predicted, we can get a more accurate prediction

            # We then check if there is a word if the corresponding POS tag among the top 10,

            best_candidate = []
            for i in range(len(most_similar_list)):
                word = most_similar_list[i][0]
                if nltk.pos_tag([word]) == pos_tag:
                    best_candidate.append(word)

            if best_candidate:  # If the list is not empty
                predicted_words.append(best_candidate[0])  # Take the first element
            else:
                predicted_words.append(most_similar_list[0][0])  # Otherwise we just take the first element

    return predicted_words
