# Function for the preprocessing
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess(document):
    """
    Do the preprocessing of the document such as removing numbers, punctuation,
    delimiting sentences, tokenizing words.
    :param document: string document
    :return: list of lines with every token words.
    """
    document = document.lower()  # Put it to lowercase
    document = document.strip()  # White space removal
    document = re.sub(r'\d+', '', document)  # Remove numbers

    # Remove all punctuation (we change the breakline in 1 before)
    document = document.replace('\n', '1')
    document = re.sub('\W+', ' ', document)
    lines = document.split('1')   # We get the different sentences.
    print("Preprocessing is finished.")

    return lines


def tokenization(lines, stemming=False, lemmatization=False):
    """
    Do the tokenization of every word, do the lemmatization and stemming
    operation as well.

    :param lines: A list of strings. Each element represents a sentence.
    :return: List of string of words. Each element represent a sentence.
    """
    stop_words = set(stopwords.words('english'))

    for i in range(len(lines)):
        lines[i] = lines[i].split()  # Tokenize the lines
        words_selected = [j for j in lines[i] if j not in stop_words]  # Remove stop words
        lines[i] = words_selected
        stemmer = PorterStemmer()  # Define the stemming operation
        lemmatizer = WordNetLemmatizer()  # Define the lemmatization
        for k in range(len(lines[i])):
            if stemming:
                lines[i][k] = stemmer.stem(lines[i][k])
            if lemmatization:
                lines[i][k] = lemmatizer.lemmatize(lines[i][k])
    print("Tokenization is finished.")
    return lines
