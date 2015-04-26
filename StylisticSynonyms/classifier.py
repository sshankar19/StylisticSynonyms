__author__ = 'Shashank'
import sys, os
import numpy as np
from scipy.sparse import csr_matrix
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
            a = fr.setup()
            b, c, d = classify(a)
            return a, b, c, d

# idf = uniqueness to classes
# tf = brute force commonalities (log tf?)


def classify(data):
    count_vect = CountVectorizer(ngram_range=(1,5))
    print type(data.train_data)
    print len(data.train_data)


    X_train_counts = count_vect.fit_transform(data.train_data)
    tfidf_transformer = TfidfTransformer()
    # print count_vect.shape
    print X_train_counts.shape
    x_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    classifier = MultinomialNB().fit(x_train_tfidf, data.cats)
    print "done with classifier"
    return count_vect, tfidf_transformer, classifier

# need to train the models on the dev set and identify how correct it is


# transforming the data_set with new data
def classify_new(style, count_vec, transformer, classifier, data):
    print type(style)
    new_counts = count_vec.transform(style)
    new_mat = transformer.transform(new_counts)
    predicted = classifier.predict(new_mat)
    print type(predicted)
    print predicted
    print len(predicted)
    print classifier.classes_
    for val in predicted:
        print data.styles[classifier.classes_[val]]


def test(data, vectorizer, classifier):
    test_docs = data.corpuses
    new_mat = vectorizer.transform(test_docs)

    predicted = classifier.predict(new_mat)

    print np.mean(predicted == data.cats)


def get_prob(style, count_vec, transformer, classifier, data):
    print type(style)
    new_counts = count_vec.transform(style)
    new_mat = transformer.transform(new_counts)
    probabilities = classifier.predict_proba(new_mat)
    prob = probabilities.tolist()
    return prob


def check_against(style, count_vec, transformer, classifier, data, the_class):
    prob = get_prob(style, count_vec, transformer, classifier)
    for probability, category in zip(prob, [0, 1, 2, 3]):
        if data.styles[category] == the_class:
            return probability
    return None


data_coll = fr.setup()
count_vec, transformer, clf = classify(data_coll)
# data_coll, count_vec, transformer, clf = load_pickled_classifier("my_classifier.pkl")
# save_pickled_classifier("my_classifier.pkl", clf)
# print "here"
# style = [data_coll.sources[3].cv_corpus[0]]
#
# # for style in data_coll.sources:
# print "Label is: "+data_coll.sources[3].label
# print get_prob(style, count_vec, transformer, clf, data_coll)
# #     print "\n"
# prediction = classify_new(style, count_vec, transformer, clf, data_coll)
# prediction = predict_new(data_coll.sources[0].cv_corpus, clf, data_coll)

print get_prob(["Gay people got dem rights too, y'know. Don't be a pushover, honey"], count_vec, transformer, clf, data_coll)
#
# save_pickled_classifier("my_classifier.pkl", clf)
# print "saved"