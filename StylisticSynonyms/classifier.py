__author__ = 'Shashank'
import cPickle

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import file_reader as fr


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
    count_vect = CountVectorizer(ngram_range=(1,1))
    X_train_counts = count_vect.fit_transform(data.train_data)
    tfidf_transformer = TfidfTransformer()
    # print count_vect.shape
    print X_train_counts.shape
    x_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    classifier = MultinomialNB().fit(x_train_tfidf, data.cats)
    print "done with classifier"
    return count_vect, tfidf_transformer, classifier

# need to train the models on the dev set and identify how correct it is
def classify_pipe(pipeline, data):
    pipeline = pipeline.fit(data.train_data, data.cats)
    return pipeline

# transforming the data_set with new data
# def classify_new(style, count_vec, transformer, classifier):
#     print type(style)
#     new_counts = count_vec.transform(style)
#     new_mat = transformer.transform(new_counts)
#     predicted = classifier.predict(new_mat)
#     print type(predicted)
#     print predicted
#     print len(predicted)
#     print classifier.classes_
#     return predicted

def classify_new_pipe(style, classifier):
    predicted = classifier.predict(style)
    return predicted

# def predict_new(style, count_vec, transformer, classifier, data):
#     predictions = classify_new(style, count_vec, transformer, classifier, data)
#     return data.styles[classifier.classes_[predictions[0]]]

def predict_new(style, pipe, data):
    predictions = classify_new_pipe(style, pipe)
    return data.styles[pipe.classes_[predictions[0]]]

# def get_prob(style, count_vec, transformer, classifier):
#     new_counts = count_vec.transform(style)
#     new_mat = transformer.transform(new_counts)
#     probabilities = classifier.predict_proba(new_mat)
#     prob = probabilities.tolist()
#     return prob

def get_prob_pipe(style, pipe):
    probabilities = pipe.predict_proba(style)
    prob = probabilities.tolist()
    return prob

# def check_against(style, count_vec, transformer, classifier, data, the_class):
#     p = get_prob(style, count_vec, transformer, classifier, data)
#     prob = p[0]
#     for probability, category in zip(prob, [0, 1, 2, 3]):
#         if data.styles[category] == the_class:
#             return probability
#     return None

def check_against_pipe(style, pipe, data):
    p = get_prob_pipe(style, pipe, data)
    prob = p[0]
    for probability, category in zip(prob, [0, 1, 2, 3]):
        if data.styles[category] == the_class:
            return probability
    return None
#
# def cv_test(count_vec, transformer, classifier, data):
#     styles = []
#     cv_corpusi = []
#     i = 0
#     for data_obj in data.sources:
#         for cv_doc in data_obj.cv_corpus:
#             styles.append(data_obj.style_code)
#             cv_corpusi.append(cv_doc)
#             i = i+1
#     print "count is: "+str(i)
#     predictions = classify_new(cv_corpusi, count_vec, transformer, classifier)
#     print np.mean(predictions == styles)

def cv_test_pipe(pipe, data):
    styles = []
    cv_corpusi = []
    i = 0
    for data_obj in data.sources:
        for cv_doc in data_obj.cv_corpus:
            styles.append(data_obj.style_code)
            cv_corpusi.append(cv_doc)
            i = i+1
    print "count is: "+str(i)
    predictions = classify_new_pipe(cv_corpusi, pipe)
    print np.mean(predictions == styles)


data_coll = fr.setup()
basic_classifier = Pipeline([('count_vec', CountVectorizer()), ('tf_idf', TfidfTransformer()), ('multi_NB', MultinomialNB())])
classify_pipe(basic_classifier, data_coll)
cv_test_pipe(basic_classifier, data_coll)

# count_vec, transformer, clf = classify(data_coll)

# cv_test(count_vec, transformer, clf, data_coll)
# data_coll, count_vec, transformer, clf = load_pickled_classifier("my_classifier.pkl")
# save_pickled_classifier("my_classifier.pkl", clf)
# print "here"
# style = [data_coll.sources[3].cv_corpus[0]]
#
# # for style in data_coll.sources:
# print "Label is: "+data_coll.sources[3].label
# print get_prob(style, count_vec, transformer, clf)
# #     print "\n"
# prediction = classify_new(style, count_vec, transformer, clf)
# prediction = predict_new(data_coll.sources[0].cv_corpus, clf, data_coll)
# #
# save_pickled_classifier("my_classifier.pkl", clf)
# print "saved"