# Here we are going to evaluate our model on the test files.
import pandas as pd
from scipy.stats import spearmanr

# We will now compute the similarity of two words


def spearman(model):
    """
    Calculate the similarities between each words and perform the Spearman ranking.
    """
    df_sim = pd.read_csv('../word_vectors/wordsim353.csv')

    # Apply the similarity function to each row
    df_test = df_sim.copy()  # Test dataframe
    df_test['Similarity'] = df_test.apply(lambda row: similarity(row['Word 1'], row['Word 2'], model),
                                          axis=1)

    spearman_rank = spearmanr(df_test['Human (mean)'], df_test['Similarity']).correlation
    df_test['Spearman'] = spearman_rank
    print("Spearman ranking between similarities is finished. Value is " + str(spearman_rank))
    df_test.to_csv('new_wordsim353.csv')
    return df_test


def similarity(word_1, word_2, model):
    """
    Compute the cosine similarity between two words.
    If the word is not known, output -1.
    :param word_1:
    :param word_2:
    :param model:
    :return:
    """
    coefficient = 0
    if word_2 in model.wv.vocab and word_1 in model.wv.vocab:
        coefficient = model.wv.similarity(word_1, word_2)
    else:
        coefficient = -1.
    return coefficient


def analogy(model):
    """
    Get the analogy for each phrase.
    :param model: The Word2Vec model which has been trained.
    :return:
    """

    df_analogy = pd.read_csv('../word_vectors/questions-words.txt', sep=" ", header=None, skiprows=1)
    df_analogy = df_analogy.dropna()  # Drop rows with NaN
    df_analogy.columns = ['Word 1', 'Analogy 1', 'Word 2', 'Analogy 2']
    df_analogy = df_analogy.applymap(lambda s: s.lower() if type(s) == str else s)  # Convert to lowercase

    df_test = df_analogy.copy()  # Get the test dataframe

    # For each line, we find the analogy and we write in the column prediction.

    df_test['Prediction'] = df_test.apply(lambda row: find_analogy(row['Word 1'], row['Analogy 1'], row['Word 2'],
                                                                   model),
                                          axis=1)
    print("Computation of the analogies is finished.")

    # Save the dataframe created
    df_test.to_csv('new_questions-word.txt')
    return df_test


def find_analogy(word_1, analogy_1, word_2, model):
    list_words = [word_1, word_2, analogy_1]
    if all(w in model.wv.vocab for w in list_words):
        word = model.most_similar(positive=[word_2, analogy_1], negative=[word_1])[0][0]
    else:
        word = 'NaN'
    return word
