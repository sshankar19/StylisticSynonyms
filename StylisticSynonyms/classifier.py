__author__ = 'Shashank'
import cPickle
from time import time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn import cross_validation

from alt_features import SentenceVariabilityVectorizer
from alt_features import SentenceLengthVectorizer
from alt_features import WordLengthVectorizer, pos_vectorizer
import file_reader as fr
from alt_features import stemmer_vectorizer
from alt_features import MyClassifier

# idf = uniqueness to classes
# tf = brute force commonalities (log tf?)

def classify(data):
    vectorizer = SentenceVariabilityVectorizer()
    t = vectorizer.transform(data.train_data)
    classifier = MultinomialNB().fit(t, data.cats)
    return count_vect, tfidf_transformer, classifier

# need to train the models on the dev set and identify how correct it is
def classify_pipe(pipeline, data):
    pipeline = pipeline.fit(data.train_data, data.cats)
    return pipeline

def classify_new_pipe(style, classifier):
    predicted = classifier.predict(style)
    return predicted

def predict_new(style, pipe, data):
    predictions = classify_new_pipe(style, pipe)
    # print predictions
    return data.styles[pipe.classes_[predictions[0]]]

def get_prob_pipe(style, pipe):
    probabilities = pipe.predict_proba(style)
    prob = probabilities.tolist()
    # print len(prob)
    return prob

def check_against_pipe(style, pipe, data, the_class):
    p = get_prob_pipe(style, pipe)
    prob = p[0]
    for probability, category in zip(prob, [0, 1, 2, 3]):
        if data.styles[category] == the_class:
            return probability
    return None

def get_styles_corpusi(data):
    styles = []
    cv_corpusi = []
    i = 0
    for data_obj in data.sources:
        for cv_doc in data_obj.cv_corpus:
            styles.append(data_obj.style_code)
            cv_corpusi.append(cv_doc)
            i = i+1
    print "count is: "+str(i)
    return (styles, cv_corpusi)

def cv_test_pipe(pipe, data):
    styles, corpusi = get_styles_corpusi(data)
    # print "a"
    predictions = classify_new_pipe(corpusi, pipe)
    # print "b"
    print np.mean(predictions == styles)

# if it's already there, don't save it again, otherwise write it.
def save_pickled_classifier(clf_fname, clf):
    try:
        with open(clf_fname) as file:
            pass
    except IOError:
        text_file = open(clf_fname, "wb")
        cPickle.dump(clf, text_file)
        text_file.close()

def save_the_data(data_fname, data):
    try:
        with open(data_fname) as file:
            pass
    except IOError:
        data_file = open(data_fname, "wb")
        cPickle.dump(data, data_file)
        data_file.close()

def load_the_data(data_fname):
    try:
        with open(data_fname, "rb") as file:
            j = cPickle.load(file)
            print "Done loading data"
            return j
    except IOError:
        print "didn't go right"
        return fr.setup()

def load_the_clf(clf_fname, data_fname):
    try:
        with open(clf_fname, "rb") as file:
            return cPickle.load(file), load_the_data(data_fname)
    except IOError:
        basic, data_coll = setup_clf(data_fname)
        return basic, data_coll



def setup_clf(data_fname):
    data_coll = load_the_data(data_fname)
    base_vect  = Pipeline([
                ('count_vec', CountVectorizer(stop_words="english")),
                ('tf_idf', TfidfTransformer()),  # list of dicts -> feature matrix
            ])
    stemma_vec = stemmer_vectorizer(0)
    trans_list = [('basic_classifier', base_vect), ('stemma_vec', stemma_vec)]
    trans_weights = { 'basic_classifier': 0.8, 'sen_var_vec': 0.1, 'stemma_vec': 0.2, 'sen_len_vec': 0.1, 'sen_word_vec': 0.1}
    # weights may not be used
    f_u = FeatureUnion(transformer_list=trans_list)
    classifier = MultinomialNB()
    pipe = Pipeline([('feature_union', f_u), ('classifier', classifier)])
    classify_pipe(pipe, data_coll)
    return pipe, data_coll


def test_feature(vectorizer, feature):
    data_coll = fr.setup()
    # classify(data_coll)
    basic_classifier = Pipeline([(feature, vectorizer), ('multi_NB', MultinomialNB())])
    classify_pipe(basic_classifier, data_coll)
    print "----TESTING " + feature+"----"
    cv_test_pipe(basic_classifier, data_coll)

def test_builtin_features():
    data_coll = fr.setup()
    # classify(data_coll)
    basic_classifier = Pipeline([('count_vec', CountVectorizer()), ('tf_idf', TfidfTransformer()), ('multi_NB', MultinomialNB())])
    classify_pipe(basic_classifier, data_coll)
    cv_test_pipe(basic_classifier, data_coll)

def unigram_clf_reg_with_mine():
    data_coll = fr.setup()
    base_vect  = Pipeline([
                ('count_vec', CountVectorizer()),
                ('tf_idf', TfidfTransformer()),  # list of dicts -> feature matrix
            ])
    sen_var_vec = SentenceVariabilityVectorizer()
    sen_len_vec = SentenceLengthVectorizer()
    sen_word_vec = WordLengthVectorizer()
    stemma_vec = stemmer_vectorizer(0)
    trans_list = [('basic_classifier', base_vect), ('sen_var_vec', sen_var_vec), ('stemma_vec', stemma_vec), ('sen_len_vec', sen_len_vec), ('sen_word_vec', sen_word_vec)]
    trans_weights = { 'basic_classifier': 0.8, 'sen_var_vec': 0.1, 'stemma_vec': 0.2, 'sen_len_vec': 0.1, 'sen_word_vec': 0.1}
    # weights may not be used
    f_u = FeatureUnion(transformer_list=trans_list)
    classifier = MultinomialNB()
    pipe = Pipeline([('feature_union', f_u), ('classifier', classifier)])
    print "Unigram clf, with sentence length and variability and word length and Stemmer,"
    classify_pipe(pipe, data_coll)
    cv_test_pipe(pipe, data_coll)

def unigram_clf_stemmer():
    data_coll = fr.setup()
    base_vect  = Pipeline([
                ('count_vec', CountVectorizer()),
                ('tf_idf', TfidfTransformer()),  # list of dicts -> feature matrix
            ])
    stemma_vec = stemmer_vectorizer(0)
    trans_list = [('basic_classifier', base_vect), ('stemma_vec', stemma_vec)]
    trans_weights = { 'basic_classifier': 0.8, 'sen_var_vec': 0.1, 'stemma_vec': 0.2, 'sen_len_vec': 0.1, 'sen_word_vec': 0.1}
    # weights may not be used
    f_u = FeatureUnion(transformer_list=trans_list)
    classifier = MultinomialNB()
    pipe = Pipeline([('feature_union', f_u), ('classifier', classifier)])
    print "Unigram clf, with only stemmer,"
    classify_pipe(pipe, data_coll)
    cv_test_pipe(pipe, data_coll)


def unigram_clf_stop_words():
    data_coll = fr.setup()
    base_vect  = Pipeline([
                ('count_vec', CountVectorizer(stop_words="english")),
                ('tf_idf', TfidfTransformer()),  # list of dicts -> feature matrix
            ])
    # stemma_vec = stemmer_vectorizer(0)
    trans_list = [('basic_classifier', base_vect)]
    trans_weights = { 'basic_classifier': 0.8, 'sen_var_vec': 0.1, 'stemma_vec': 0.2, 'sen_len_vec': 0.1, 'sen_word_vec': 0.1}
    # weights may not be used
    f_u = FeatureUnion(transformer_list=trans_list)
    classifier = MultinomialNB()
    pipe = Pipeline([('feature_union', f_u), ('classifier', classifier)])
    print "Unigram clf with just stop words,"
    classify_pipe(pipe, data_coll)
    cv_test_pipe(pipe, data_coll)

def n_grams_test():
    data_coll = fr.setup()
    base_vect  = Pipeline([
                ('count_vec', CountVectorizer(ngram_range=(1,5))),
                ('tf_idf', TfidfTransformer()),  # list of dicts -> feature matrix
            ])
    # stemma_vec = stemmer_vectorizer(0)
    trans_list = [('basic_classifier', base_vect)]
    trans_weights = { 'basic_classifier': 0.8, 'sen_var_vec': 0.1, 'stemma_vec': 0.2, 'sen_len_vec': 0.1, 'sen_word_vec': 0.1}
    # weights may not be used
    f_u = FeatureUnion(transformer_list=trans_list)
    classifier = MultinomialNB()
    pipe = Pipeline([('feature_union', f_u), ('classifier', classifier)])
    print "JUST ngram clf,"
    classify_pipe(pipe, data_coll)
    cv_test_pipe(pipe, data_coll)

def unigram_baseline_test():
    data_coll = fr.setup()
    base_vect  = Pipeline([
                ('count_vec', CountVectorizer()),
                ('tf_idf', TfidfTransformer()),  # list of dicts -> feature matrix
            ])
    # stemma_vec = stemmer_vectorizer(0)
    trans_list = [('basic_classifier', base_vect)]
    trans_weights = { 'basic_classifier': 0.8, 'sen_var_vec': 0.1, 'stemma_vec': 0.2, 'sen_len_vec': 0.1, 'sen_word_vec': 0.1}
    # weights may not be used
    f_u = FeatureUnion(transformer_list=trans_list)
    classifier = MultinomialNB()
    pipe = Pipeline([('feature_union', f_u), ('classifier', classifier)])
    print "base unigram clf"
    classify_pipe(pipe, data_coll)
    print type(pipe.classes_)
    print pipe.classes_
    cv_test_pipe(pipe, data_coll)

def test_framework(n_grams, stop_w, stemmer, sentence_l, sentence_v, word_l):
    data_coll = fr.setup()
    classifier_string = ""
    if n_grams == 0 or n_grams == 1:
        if stop_w != 0:
            base_vect = Pipeline([ ('count_vec', CountVectorizer(stop_words="english")), ('tf_idf', TfidfTransformer()), ])
            classifier_string += "base unigram clf with english stop words (not in stemmer) "
        else:
            base_vect = Pipeline([ ('count_vec', CountVectorizer()), ('tf_idf', TfidfTransformer()), ])
            classifier_string += "base unigram clf without stop words "
    else:
        if stop_w != 0:
            base_vect = Pipeline([ ('count_vec', CountVectorizer(ngram_range=(1, n_grams), stop_words="english")), ('tf_idf', TfidfTransformer()), ])
            classifier_string += "n-gram clf (1 to "+str(n_grams)+" grams) with english stop words (not in stemmer) "
        else:
            base_vect = Pipeline([ ('count_vec', CountVectorizer(ngram_range=(1, n_grams))), ('tf_idf', TfidfTransformer()), ])
            classifier_string += "n-gram clf (1 to "+str(n_grams)+" grams) without stop words "
    trans_list = [('base_vect', base_vect)]
    if stemmer != 0:
        trans_list.append(('stemma_vec', stemmer_vectorizer(0)))
        classifier_string += " with stemming "
    else:
        classifier_string += " withOUT stemming "
    if sentence_l != 0:
        trans_list.append(('sent_length', SentenceLengthVectorizer()))
        classifier_string += " with sentence length "
    else:
        classifier_string += " withOUT sentence lenght "
    if sentence_v != 0:
        trans_list.append(('sent_var', SentenceVariabilityVectorizer()))
        classifier_string += " with sentence variability "
    else:
        classifier_string += " withOUT sentence variability "
    if word_l != 0:
        trans_list.append(('word_length', WordLengthVectorizer()))
        classifier_string += " with word length "
    else:
        classifier_string += " withOUT word length "
    f_u = FeatureUnion(transformer_list=trans_list)
    classifier = MultinomialNB()
    pipe = Pipeline([('feature_union', f_u), ('classifier', classifier)])
    print classifier_string
    classify_pipe(pipe, data_coll)
    cv_test_pipe(pipe, data_coll)

def cross_val():
    data_coll = fr.setup()
    base_vect  = Pipeline([
                ('count_vec', CountVectorizer(ngram_range=(1,1), max_features=50)),
                ('tf_idf', TfidfTransformer()),  # list of dicts -> feature matrix
            ])
    stemma_vec = stemmer_vectorizer(0)
    trans_list = [('basic_classifier', base_vect), ('stemma_vec', stemma_vec)]
    trans_weights = { 'basic_classifier': 0.8, 'sen_var_vec': 0.1, 'stemma_vec': 0.2, 'sen_len_vec': 0.1, 'sen_word_vec': 0.1}
    # weights may not be used
    f_u = FeatureUnion(transformer_list=trans_list)
    classifier = MultinomialNB()
    pipe = Pipeline([('feature_union', f_u), ('classifier', classifier)])
    print "n_gram clf, withOUT sentence and variability and word,"
    styles, corpusi = get_styles_corpusi(data_coll)
    k = cross_validation.cross_val_score(pipe, corpusi, styles, cv=5, verbose=True, n_jobs=4)
    print k

def my_features_test():
    data_coll = fr.setup()
    base_vect  = Pipeline([
                ('count_vec', SentenceVariabilityVectorizer()),
            ])
    # stemma_vec = stemmer_vectorizer(0)
    trans_list = [('basic_classifier', base_vect)]
    trans_weights = { 'basic_classifier': 0.8, 'sen_var_vec': 0.1, 'stemma_vec': 0.2, 'sen_len_vec': 0.1, 'sen_word_vec': 0.1}
    # weights may not be used
    f_u = FeatureUnion(transformer_list=trans_list)
    classifier = MyClassifier()
    pipe = Pipeline([('feature_union', f_u), ('classifier', classifier)])
    print "word length clf"
    classify_pipe(pipe, data_coll)
    cv_test_pipe(pipe, data_coll)
    # print "nada"

def just_part_of_speech_trigrams():
    data_coll = fr.setup()
    pos_vect = pos_vectorizer()
    trans_list = [('pos_vect', pos_vect)]
    f_u = FeatureUnion(transformer_list=trans_list)
    classifier = MultinomialNB()
    pipe = Pipeline([('feature_union', f_u), ('classifier', classifier)])
    print "part of speech trigrams only"
    classify_pipe(pipe, data_coll)
    cv_test_pipe(pipe, data_coll)

# stop words, stemming, unigram, sublinear tf = true, use_idf = false
def test_best_features():
    data_coll = fr.setup()
    base_vect  = Pipeline([
                ('count_vec', CountVectorizer(stop_words="english", ngram_range=(1,1))),
                ('tf_idf', TfidfTransformer(use_idf=False)),  # list of dicts -> feature matrix
                ])
    stemma_vec = stemmer_vectorizer(0)
    trans_list = [('basic_classifier', base_vect), ('stemma_vec', stemma_vec)]
    f_u = FeatureUnion(transformer_list=trans_list)
    classifier = MultinomialNB()
    pipe = Pipeline([('feature_union', f_u), ('classifier', classifier)])
    print "Best features: stop words, stemmings, unigram, sublinear_tf=False, use_idf=false"
    classify_pipe(pipe, data_coll)
    cv_test_pipe(pipe, data_coll)

def save_best():
    data_coll = fr.setup()
    print "Done Setting up"
    base_vect  = Pipeline([
                ('count_vec', CountVectorizer(stop_words="english", ngram_range=(1,3))),
                ('tf_idf', TfidfTransformer()),  # list of dicts -> feature matrix
                ])
    stemma_vec = stemmer_vectorizer(0)
    trans_list = [('basic_classifier', base_vect), ('stemma_vec', stemma_vec)]
    f_u = FeatureUnion(transformer_list=trans_list)
    classifier = MultinomialNB()
    pipe = Pipeline([('feature_union', f_u), ('classifier', classifier)])
    pipe = classify_pipe(pipe, data_coll)
    print "Done with classification"
    save_the_data("data.pkl", data_coll)
    print "Done Saving the data"
    save_pickled_classifier("best.pkl", pipe)
    print "Done saving the classifier"

def test_loaded_classifier():
    t0 = time()
    classifer, data = load_the_clf("best.pkl", "data.pkl")
    print("done1 in %0.3fs" % (time() - t0))
    print "loaded the classifier+data"
    cv_test_pipe(classifer, data)
    print("done2 in %0.3fs" % (time() - t0))
    print "done with test"

# unigram_baseline_test()
# could not save best took up too many resources
# save_best()
# test_loaded_classifier()
# test_save()
# test_best_features()
# Base
# test_framework(0, 0, 0, 0, 0, 0)
# # up to trigrams
# test_framework(3, 0, 0, 0, 0, 0)
# # Base with stop words
# test_framework(0, 1, 0, 0, 0, 0)
# # Base with stemming (no stop words)
# test_framework(0, 0, 1, 0, 0, 0)
# # Base with those 3 features
# test_framework(0, 1, 1, 0, 0, 0)
# # N grams with those 3 features
# test_framework(3, 1, 1, 0, 0, 0)
# up to 5 grams
# test_framework(5, 0, 0, 0, 0, 0)
# 5 grams with those 3 features
# test_framework(5, 1, 1, 0, 0, 0)

# just_part_of_speech_trigrams()
# unigram_baseline_test()
# my_features_test()

# def test_save():
#     try:
#         with open("try.pkl") as file:
#             print "here2"
#             pass
#     except IOError:
#         text_file = open("try.pkl", "wb")
#         print "here"
#         cPickle.dump("LFDSJLFDSJL:FDSaafksavghalajflaeiokl", text_file)
#         text_file.close()