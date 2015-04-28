__author__ = 'Shashank'

from sklearn.base import BaseEstimator
from nltk import word_tokenize
from nltk import sent_tokenize
import numpy as np
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk


def extend(a_list, num_to_extend):
    a_list += [0] * (num_to_extend - len(a_list))
    return a_list


class SentenceVariabilityVectorizer(BaseEstimator):

    def fit(self, documents, y=None):
        return self

    # documents is a list of strings
    def transform(self, documents):
        # lists of doc_vectors
        freq_lists = []

        variability_lists = []
        max_len = 0
        for doc in documents:
            doc_freq = []
            dict_values = {}
            doc = unicode(doc, 'utf-8')
            sentence_list = sent_tokenize(doc)
            number_of_sentences = len(sentence_list)
            length = len(max(sentence_list, key=len))
            # for each value of sentence length possible in this doc, initialize to 0
            for i in range(length+1):
                dict_values[i] = 0

            # set the max length
            if length > max_len:
                max_len = length

            # iterate over all sentences in list insert value into doc_vector
            for sentence in sentence_list:
                temp_length = len(sentence)
                dict_values[temp_length] += 1

            # make frequencies for them, by dividing by number of sentences for each value in dictionary
            for key in dict_values:
                dict_values[key] /= float(number_of_sentences)

            # making this into a list (which I know is in order)
            for i in range(length+1):
                doc_freq.append(dict_values[i])

            freq_lists.append(doc_freq)

        # documents should be in order as well, since it's been iterated over
        for doc_freq in freq_lists:
            freq_length = len(doc_freq)
            if freq_length < max_len:
                doc_freq = extend(doc_freq, max_len)

        # Now find all the documents where there is a value greater than 0, and sum up and divide by total

        for doc in freq_lists:
            num_sentences_doc = 0
            sum = 0.0
            for d in doc:
                if d > 0:
                    sum += d
                num_sentences_doc += 1
            variability_lists.append([sum/float(num_sentences_doc)])

        # print variability_lists
        results = np.array(variability_lists)
        # print results.__len__()
        # print results
        return results


class SentenceLengthVectorizer(BaseEstimator):

    def fit(self, documents, y=None):
        return self

    # documents is a list of strings
    def transform(self, documents):
        average_len = []
        for doc in documents:
            doc = unicode(doc, 'utf-8')
            sentence_list = sent_tokenize(doc)

            number_of_sentences = len(sentence_list)
            # print type(sentence_list[0])

            average = 0.0
            for sentence in sentence_list:
                temp_length = len(sentence)
                average += temp_length

            average /= float(number_of_sentences)

            average_len.append([average])

        results = np.array(average_len)
        # print results
        return results

class WordLengthVectorizer(BaseEstimator):

    def fit(self, documents, y=None):
        return self

    # documents is a list of strings
    def transform(self, documents):
        average_len = []
        for doc in documents:
            doc = unicode(doc, 'utf-8')
            word_list = word_tokenize(doc)
            number_of_word = len(word_list)
            average = 0.0
            for word in word_list:
                temp_length = len(word)
                average += temp_length

            average /= float(number_of_word)

            average_len.append([average])

        results = np.array(average_len)
        return results

# Based off of http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html

stemmer = PorterStemmer()


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


def stemmer_vectorizer(stop_words):
    if stop_words == 0:
        return TfidfVectorizer(tokenizer=tokenize)
    else:
        return TfidfVectorizer(tokenizer=tokenize, stop_words='english')