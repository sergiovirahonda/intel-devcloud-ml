# Python imports
import logging
import os
import re
import sys

# ML-related imports
import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()
import gensim
import pandas as pd
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Logger definition
logger = logging.getLogger(__name__)
log_formatter = logging.Formatter(
    '{"timestamp":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}', 
    "%Y-%m-%dT%H:%M:%S"
)
LOG_FILE = f"{os.getcwd()}/logs/train.log"
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# Data importing
logger.info('Importing datasets...')
try:
    train_dataset = pd.read_csv(f'{os.getcwd()}/datasets/train.csv')
    test_dataset = pd.read_csv(f'{os.getcwd()}/datasets/test.csv')
    logger.info(f'Datasets imported.')
    logger.info(f'Total records for training: {len(train_dataset)} ')
    logger.info(f'Total records for testing: {len(test_dataset)} ')
except Exception as e:
    logger.error(f'Datasets could not be loaded: {e}')
    sys.exit(1)
    
# Train dataset cleaning
logger.info('Cleaning training dataset...')
train_dataset = train_dataset[['selected_text','sentiment']]
train_dataset.dropna(inplace=True)
train_dataset.reset_index(inplace=True, drop=True)

def clean_sentence(sentence: str) -> str:

    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    _sentence = url_pattern.sub(r'', sentence) # Removed URLs
    _sentence = re.sub('\S*@\S*\s?', '', _sentence) # Removed emails
    _sentence = re.sub('\s+', ' ', _sentence) # Removed space chars
    _sentence = re.sub("\'", "", _sentence) # Removed backslashes
    _sentence = gensim.utils.simple_preprocess(str(_sentence), deacc=True)
    processed_sentence = TreebankWordDetokenizer().detokenize(_sentence)
    return processed_sentence

for index, row in train_dataset.iterrows():
    sentence = train_dataset.iloc[index]['selected_text']
    processed_sentence = clean_sentence(sentence)
    train_dataset.at[index,'selected_text'] = processed_sentence
    
# Test dataset cleaning
logger.info('Cleaning testing dataset...')
test_dataset = test_dataset[['text','sentiment']]
test_dataset.dropna(inplace=True)
test_dataset.reset_index(inplace=True, drop=True)

for index, row in test_dataset.iterrows():
    sentence = test_dataset.iloc[index]['text']
    processed_sentence = clean_sentence(sentence)
    test_dataset.at[index,'text'] = processed_sentence

# Label encoding
logger.info('Encoding labels...')
label_encoder = LabelEncoder()
train_dataset['sentiment'] = label_encoder.fit_transform(
    train_dataset['sentiment']
    )

classes = label_encoder.classes_

test_dataset['sentiment'] = label_encoder.transform(
    test_dataset['sentiment']
    )

# Vectorizing words with TFIDF approach
logger.info('Vectorizing training dataset...')
vectorizer = TfidfVectorizer(min_df=3, ngram_range=(1,5))
vectorizer.fit(
    train_dataset['selected_text']
)
vectorized_train_dataset = vectorizer.transform(
    train_dataset['selected_text'])

vectorized_test_dataset = vectorizer.transform(
    test_dataset['text'])

# Classifier definition
logger.info('Training Support Vector Machine classifier...')
classifier = SVC(kernel='linear', C=1)
classifier.fit(
    vectorized_train_dataset,
    train_dataset['sentiment'],
    )
logger.info('Classifier trained.')

# Predicting labels for test dataset
logger.info('Predicting test labels...')
predictions = classifier.predict(vectorized_test_dataset)
predictions = list(predictions)
target = list(test_dataset['sentiment'])

classifier_score = classifier.score(
    vectorized_test_dataset,
    test_dataset['sentiment']
    )
logger.info(f'Score achieved: {classifier_score}')

_conf_matrix = confusion_matrix(target, predictions)
conf_matrix = pd.DataFrame(
    _conf_matrix, 
    index = classes,
    columns = classes)
conf_matrix = conf_matrix.astype('float') /\
    conf_matrix.sum(axis=1)[:, np.newaxis]
logger.info(f'Confusion matrix :\n {conf_matrix}')