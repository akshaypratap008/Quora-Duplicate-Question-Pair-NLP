import os
from bs4 import BeautifulSoup
import re
import distance
from fuzzywuzzy import fuzz
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import pickle
import numpy as np
from scipy.sparse import hstack, csr_matrix
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP_WORDS = set(ENGLISH_STOP_WORDS)

def preprocess(question: str) -> str:
    question = str(question).lower().strip()
    question = question.replace('%', ' percent ')
    question = question.replace('$', ' dollar ')
    question = question.replace('₹', ' rupee ')
    question = question.replace('€', ' euro ')
    question = question.replace('@', ' at ')
    question = question.replace('[math]', '')
    question = question.replace(',000,000,000 ', ' b ')
    question = question.replace(',000,000 ', ' m ')
    question = question.replace(',000 ', ' k ')
    question = re.sub(r'([0-9]+)000000000', r'\1b', question)
    question = re.sub(r'([0-9]+)000000', r'\1m', question)
    question = re.sub(r'([0-9]+)000', r'\1k', question)
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "can not",
        "can't've": "can not have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have",
    }
    words = []
    for word in question.split():
        if word in contractions:
            word = contractions[word]
        words.append(word)
    question = ' '.join(words)
    question = question.replace("'ve", ' have')
    question = question.replace("n't", ' not')
    question = question.replace("'re", ' are')
    question = question.replace("'ll", ' will')
    question = question.replace("n't", ' not')
    question = BeautifulSoup(question, 'html.parser').get_text()
    question = re.sub(r'\W', ' ', question).strip()
    return question

def fetch_extra_features(question1: str, question2: str) -> np.array:
    q1 = preprocess(question1)
    q2 = preprocess(question2)

    q1_len = len(q1)
    q2_len = len(q2)
    q1_words = len(q1.split())
    q2_words = len(q2.split())
    combined = q1 + ' ' + q2
    words_unique = len(set(combined.split()))
    words_total = len(combined)
    words_common = len(set(q1.split()) & set(q2.split())) 

    extra_features = {
        'q1_len': q1_len,
        'q2_len': q2_len,
        'q1_words': q1_words,
        'q2_words': q2_words,
        'words_unique': words_unique,
        'words_total': words_total,
        'words_common': words_common
        }
    
    return np.array(list(extra_features.values()))

def fetch_advanced_features(question1: str, question2: str) -> np.array: 
    
    q1_tokens = question1.split()
    q2_tokens = question2.split()

    if not q1_tokens or not q2_tokens:
        token_features = [0.0] * 8
    else:
        q1_non_stops = set([token for token in q1_tokens if token not in STOP_WORDS])
        q2_non_stops = set([token for token in q2_tokens if token not in STOP_WORDS])
        q1_stops = set([token for token in q1_tokens if token in STOP_WORDS])
        q2_stops = set([token for token in q2_tokens if token in STOP_WORDS])
        common_words_count = len(q1_non_stops.intersection(q2_non_stops))
        common_stop_count = len(q1_stops.intersection(q2_stops))
        common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

        SAFE_DIV = 1e-4

        'cwc_min', 'cwc_max', 'csc_min', 'csc_max', 'ctc_min', 'ctc_max',
        'last_word_eq', 'first_word_eq', 'abs_len_diff', 'mean_len', 'longest_substr_ratio',
        'fuzz_ratio', 'fuzz_partial_ratio', 'token_sort_ratio', 'token_set_ratio'


        # token features 
        cwc_min = common_words_count / (min(len(q1_non_stops), len(q2_non_stops)) + SAFE_DIV)       
        cwc_max = common_words_count / (max(len(q1_non_stops), len(q2_non_stops)) + SAFE_DIV)
        csc_min = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
        csc_max = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
        ctc_min = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        ctc_max = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        last_word_eq = int(q1_tokens[-1] == q2_tokens[-1])
        first_word_eq = int(q1_tokens[0] == q2_tokens[0])

    # length based features
    if not q1_tokens or not q2_tokens:
        abs_len_diff = 0.0
        mean_len = 0.0
        longest_substr_ratio = 0.0
    else:
        abs_len_diff = abs(len(q1_tokens) - len(q2_tokens))
        mean_len = (len(q1_tokens) + len(q2_tokens)) / 2.0
        common_substrings = list(distance.lcsubstrings(question1, question2))
        if common_substrings:
            longest_substr_ratio = len(common_substrings[0]) / (min(len(question1), len(question2)) + 1)
        else:
            longest_substr_ratio = 0.0

    # fuzzy features
    fuzz_ratio = fuzz.ratio(question1, question2)
    fuzz_partial_ratio = fuzz.partial_ratio(question1, question2)
    token_sort_ratio = fuzz.token_sort_ratio(question1, question2)
    token_set_ratio = fuzz.token_set_ratio(question1, question2)

    advanced_features = {
        'cwc_min': cwc_min,
        'cwc_max':cwc_max,
        'csc_min': csc_min,
        'csc_max': csc_max,
        'ctc_min': ctc_min,
        'ctc_max': ctc_max, 
        'last_word_eq': last_word_eq,
        'first_word_eq': first_word_eq,
        'abs_len_diff': abs_len_diff,
        'mean_len': mean_len,
        'longest_substr_ratio': longest_substr_ratio,
        'fuzz_ratio': fuzz_ratio,
        'fuzz_partial_ratio': fuzz_partial_ratio,
        'token_sort_ratio': token_sort_ratio,
        'token_set_ratio': token_set_ratio
    }

    return np.array(list(advanced_features.values()))

def fetch_additional_features(question1: str, question2: str) -> np.array:
    extra_features_arr = fetch_extra_features(question1, question2)
    advanced_features_arr = fetch_advanced_features(question1, question2)

    additional_features_arr = np.hstack((extra_features_arr, advanced_features_arr)).reshape(1,-1)

    # standard scalling
    with open(os.path.join(BASE_DIR, 'artifacts', 'scaler.pkl'), 'rb') as scaler_file:
        scaler_obj = pickle.load(scaler_file)
    
    additional_features_arr_scaled = scaler_obj.transform(additional_features_arr)

    return additional_features_arr_scaled
    
def text_vectorization(question1: str, question2:str):      # will return sparse array
    
    # load vectorizer pickle file 
    with open(os.path.join(BASE_DIR, 'artifacts', 'tfidf_vectorizer.pkl'), 'rb') as file:
        vectorizer = pickle.load(file)

    q1_vec = vectorizer.transform([question1])
    q2_vec = vectorizer.transform([question2])

    features_vec = hstack([q1_vec, q2_vec,abs(q1_vec - q2_vec)])

    return features_vec

def preprocessing_pipeline(question1: str, question2:str):      # returns sparse array
    question1 = preprocess(question1)
    question2 = preprocess(question2)

    additional_features_scaled_dense = fetch_additional_features(question1, question2)
    adiitional_features_scaled_sparse = sparse.csc_matrix(additional_features_scaled_dense)
    text_vec = text_vectorization(question1, question2)

    final_features = hstack([text_vec, adiitional_features_scaled_sparse])

    return final_features

def prediction_pipeline(question1: str, question2:str):
    input_transformed_features_arr = preprocessing_pipeline(question1, question2)

    with open(os.path.join(BASE_DIR, 'artifacts', 'model.pkl'), 'rb') as file:
        model = pickle.load(file)

    prediction = model.predict(input_transformed_features_arr)
    probability = model.predict_proba(input_transformed_features_arr)

    return prediction, probability
    

question1 = 'Hello how are you?'
question2 = 'I am good thank you'

pred, prob = prediction_pipeline(question1, question2)
print('pred: ', pred)
print('prob:', prob)

# print(fetch_additional_features(question1, question2).reshape(1, -1))





    

