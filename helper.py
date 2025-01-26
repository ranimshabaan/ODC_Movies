# import pickle
# import re
# import nltk 
# import html
# import pandas as pd
# import numpy as np
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize
# import unicodedata
# import string
# from sklearn.feature_extraction.text import TfidfVectorizer




# ##download the necessary resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('punkt_tab')
# nltk.download('wordnet')

# stop_words = stopwords.words('english')
# lemmatizer = WordNetLemmatizer()

# ## load the vectorizer
# tf = pickle.load(open("artifacts/tf.pkl", 'rb'))


# def remove_special_chars(text):
#     re1 = re.compile(r'  +')
#     x1 = text.lower().replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
#         'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
#         '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
#         ' @-@ ', '-').replace('\\', ' \\ ')
#     return re1.sub(' ', html.unescape(x1))


# def remove_non_ascii(text):
#     """Remove non-ASCII characters from list of tokenized words"""
#     return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')


# def to_lowercase(text):
#     return text.lower()



# def remove_punctuation(text):
#     """Remove punctuation from list of tokenized words"""
#     translator = str.maketrans('', '', string.punctuation)
#     return text.translate(translator)


# def replace_numbers(text):
#     """Replace all interger occurrences in list of tokenized words with textual representation"""
#     return re.sub(r'\d+', '', text)


# def remove_whitespaces(text):
#     return text.strip()


# def remove_stopwords(words, stop_words):
#     """
#     :param words:
#     :type words:
#     :param stop_words: from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
#     or
#     from spacy.lang.en.stop_words import STOP_WORDS
#     :type stop_words:
#     :return:
#     :rtype:
#     """
#     return [word for word in words if word not in stop_words]


# def lemmatize_words(words):
#     """Lemmatize words in text"""

#     lemmatizer = WordNetLemmatizer()
#     return [lemmatizer.lemmatize(word) for word in words]

# def lemmatize_verbs(words):
#     """Lemmatize verbs in text"""

#     lemmatizer = WordNetLemmatizer()
#     return ' '.join([lemmatizer.lemmatize(word, pos='v') for word in words])

# def text2words(text):
#   return word_tokenize(text)

# def normalize_text( text):
#     text = remove_special_chars(text)
#     text = remove_non_ascii(text)
#     text = remove_punctuation(text)
#     text = to_lowercase(text)
#     text = replace_numbers(text)
#     words = text2words(text)
#     words = remove_stopwords(words, stop_words)
#     # words = stem_words(words)# Either stem ovocar lemmatize
#     words = lemmatize_words(words)
#     words = lemmatize_verbs(words)

#     return ''.join(words)


import pickle
import re
import nltk
import html
import unicodedata
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stop words and lemmatizer
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# Load the TF-IDF vectorizer
tf = pickle.load(open("artifacts/tf.pkl", 'rb'))

def remove_special_chars(text):
    """Remove special characters and HTML entities."""
    re1 = re.compile(r'  +')
    x1 = text.lower().replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x1))

def remove_non_ascii(text):
    """Remove non-ASCII characters."""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

def to_lowercase(text):
    """Convert text to lowercase."""
    return text.lower()

def remove_punctuation(text):
    """Remove punctuation from text."""
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def replace_numbers(text):
    """Remove numbers from the text."""
    return re.sub(r'\d+', '', text)

def remove_whitespaces(text):
    """Remove leading and trailing whitespaces."""
    return text.strip()

def remove_stopwords(words):
    """Remove stop words from the list of tokenized words."""
    return [word for word in words if word not in stop_words]

def lemmatize_words(words):
    """Lemmatize words."""
    return [lemmatizer.lemmatize(word) for word in words]

def text2words(text):
    """Tokenize text into words."""
    return word_tokenize(text)

def normalize_text(text):
    """Normalize and preprocess text."""
    text = remove_special_chars(text)
    text = remove_non_ascii(text)
    text = remove_punctuation(text)
    text = to_lowercase(text)
    text = replace_numbers(text)
    words = text2words(text)
    words = remove_stopwords(words)
    words = lemmatize_words(words)
    return ' '.join(words)  # Return text as a single string
