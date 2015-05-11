__author__ = 'Shashank'

from nltk import sent_tokenize, pos_tag, word_tokenize
import numpy as np
from scipy import sparse
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn.base import BaseEstimator, ClassifierMixin

# class EnsembleClassifier(BaseEstimator, ClassifierMixin):
#     def __init__(self, classifiers=None):
#         self.classifiers = classifiers
#
#     def fit(self, X, y):
#         for classifier in self.classifiers:
#             classifier.fit(X, y)
#
#     def predict_proba(self, X):
#         self.predictions_ = list()
#         for classifier in self.classifiers:
#             self.predictions_.append(classifier.predict_proba(X))
#         return np.mean(self.predictions_, axis=0)


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
        # print documents
        average_len = []
        for doc in documents:
            doc = unicode(doc, 'utf-8')
            word_list = word_tokenize(doc)
            number_of_word = len(word_list)
            # print "\n"
            # print "Doc is: " + doc
            # print word_list[:20]
            # print "\n"
            average = 0.0
            for word in word_list:
                temp_length = len(word)
                average += temp_length

            # print "sum is "+ str(average)
            average /= float(number_of_word)
            # print "number of words is: " + str(number_of_word)
            # print "average is " + str(average)
            average_len.append([average])

        results = np.array(average_len)

        b = sparse.csr_matrix(results)
        # print results
        # print b
        return b

# Stemmer is from http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html

stemmer = PorterStemmer()


def stem_tokens(tokens, stemmer):
    stems = []
    for item in tokens:
        stems.append(stemmer.stem(item))
    return stems


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


def stemmer_vectorizer(stop_words):
    if stop_words == 0:
        return TfidfVectorizer(tokenizer=tokenize)
    else:
        return TfidfVectorizer(tokenizer=tokenize, stop_words='english')


# part of speech tagger
# tfidf vectorizer (tri gram) trained on the part of speech
def convert_to_pos(the_string):
    text = word_tokenize(the_string)
    tuple_list = pos_tag(text)
    my_list = [x[1] for x in tuple_list]
    new_string = " ".join(str(x) for x in my_list)
    return new_string


def pos_vectorizer():
    return TfidfVectorizer(tokenizer=convert_to_pos, ngram_range=(3, 3))


def find_min(number, dictionary):
    return min(dictionary.items(), key=lambda x: abs(x[1]-number))

# noinspection PyAttributeOutsideInit
class MyClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.classes_ = np.unique([0, 1, 2, 3])
        new_list = X.tolist()
        self.styles = [1, 2, 2, 3, 3, 0, 1, 0]
        self.data = [item[0] for item in new_list]
        dictionary = {}
        self.averages = []
        for style, item in zip(self.styles, self.data):
            dictionary[style] = dictionary.get(style, 0) + item
        for key in dictionary:
            dictionary[key] /= 2.0
            self.averages.append(dictionary[key])
        self.dictionary = dictionary
        return self

    def predict(self, X):
        new_list = X.tolist()
        mins = []
        for stupid in new_list:
            item = stupid[0]
            minum = find_min(item, self.dictionary)
            mins.append(minum[0])
        mins2 = np.array(mins)
        return mins2


def classify(data):
    vectorizer = SentenceLengthVectorizer()
    t = vectorizer.transform(data.train_data)
    classifier = MyClassifier()
    classifier = classifier.fit(t, data.cats)
    print type(classifier.classes_)
    print classifier.classes_
    new_x = vectorizer.transform([data.sources[0].cv_corpus[0]])
    print classifier.predict(new_x)
    print "done with classifier"

# data = fr.setup()
# classify(data)
# basic_classifier = Pipeline([( "Stemmer Vectorizer", SentenceLengthVectorizer()), ('multi_NB', MultinomialNB())])
# basic = classify_pipe(basic_classifier, data)