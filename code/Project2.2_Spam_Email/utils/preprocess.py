import re
import unidecode
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def lowercase_text(text):
    """Convert text to lowercase."""
    return text.lower()

def remove_special_characters(text):
    """Remove special characters from text."""
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def tokenize_text(text):
    """Tokenize text into words."""
    return word_tokenize(text)

stopwords = set(stopwords.words('english'))

def remove_stopwords(tokens, stopwords):
    """Remove stopwords from tokenized text."""
    return [token for token in tokens if token not in stopwords]

def unidecode_text(text):
    """Convert text to ASCII using unidecode."""
    return unidecode.unidecode(text)

def lemmatize_text(tokens):
    """Lemmatize tokens."""
    lemmatizer = nltk.WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def remove_rare_words(tokens, min_freq=2):
    """Remove rare words from tokens."""
    freq_dist = nltk.FreqDist(tokens)
    return [token for token in tokens if freq_dist[token] >= min_freq]

def negation_handling(tokens):
    """Handle negation in text."""
    negation_words = {'not', 'no', 'never', 'nothing', 'nowhere', 'neither', 'nobody', 'nowhere',
                      'cannot', "can't", "won't", "isn't", "aren't", "wasn't", "weren't"}

    punct = {'!', '.', '?'}
    negated_token = False
    processed_tokens = []

    for toke in tokens:
        if toke in negation_words:
            negated_token = True
            processed_tokens.append(toke)
        elif toke in punct:
            negated_token = False
            processed_tokens.append(toke)
        elif negated_token:
            processed_tokens.append('neg_' + toke)
        else:
            processed_tokens.append(toke)
    return processed_tokens

def preprocess_text(text):
    """Preprocess text by apply all steps."""
    text = lowercase_text(text)
    text = unidecode_text(text)
    text = remove_special_characters(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens, stopwords)
    tokens = lemmatize_text(tokens)
#    tokens = remove_rare_words(tokens)
#    tokens = negation_handling(tokens)
    return ' '.join(tokens)