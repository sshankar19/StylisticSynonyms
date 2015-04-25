__author__ = 'Shashank'
import sys, os
import numpy as np
import cPickle
import file_reader as fr
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def save_pickled_classifier(clf_fname, clf):
    try:
        with open(clf_fname) as file:
            pass
    except IOError:
        text_file = open(clf_fname, "wb")
        cPickle.dump(clf, text_file)
        text_file.close()


def load_pickled_classifier(clf_fname):
    try:
        with open(clf_fname) as file:
            return cPickle.load(file)
    except IOError:
        print "Sorry we couldn't find the file?"
        return None

# idf = uniqueness to classes
# tf = brute force commonalities (log tf?)


def classify(data):
    # Do I want inverse document frequency? No right? Because the words usage themselves are different
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(data.train_data)
    print X_train_counts.shape
    # print count_vect.vocabulary_
    print count_vect.vocabulary_.get(u'algorithm')
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    print x_train_tfidf


    # vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
    # print "done with vectorizer"
    # print vectorizer.get_feature_names()
    # print vectorizer
    # tfidfmatrix = vectorizer.fit_transform(data.train_data)
    # print "done with tfidfmatrix"
    # print tfidfmatrix
    # classifier = MultinomialNB().fit(tfidfmatrix, data.cats)
    # print "done with classifier"
    # return vectorizer, tfidfmatrix, classifier
    return count_vect, None, None

# need to train the models on the dev set and identify how correct it is


def predict_new(style, classifier, data):
    prediction = classifier.predict(style)
    print prediction
    print classifier.classes_
    return data.styles[prediction]


# transforming the data_set with new data
def classify_new(style, vectorizer, classifier):

    new_mat = vectorizer.transform(style)
    predicted = classifier.predict(new_mat)

    for val in predicted:
        print(val)

    return val


def test(data, vectorizer, classifier):
    test_docs = data.corpuses
    new_mat = vectorizer.transform(test_docs)

    predicted = classifier.predict(new_mat)

    print np.mean(predicted == data.cats)


def get_prob(style, data, classifier, the_class):
    probabilities = classifier.predict_proba(style)
    prob = probabilities.tolist()
    for probability, category in zip(prob, [0, 1, 2, 3]):
        if data.styles[category] == the_class:
            return probability
    return None

data_coll = fr.setup()
vec, tfidf, clf = classify(data_coll)
# print "here"
#
# save_pickled_classifier("my_classifier.pkl", clf)
# print "saved"