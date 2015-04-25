__author__ = 'Shashank'

import sys, os


class data_sets(object):
    pass


class data_collection(object):
    pass


def put_in_string(path):
    string = ""
    for folder, sub_folders, documents in os.walk(path):
        for each_file in documents:
            k = open(path+each_file)
            for line in k:
                string += line
            k.close()
    return string


def initialize_data(data_obj, root, folder_list):
    cv_path = root+data_obj.label+"/"+folder_list[0]
    test_path = root+data_obj.label+"/"+folder_list[1]
    training_path = root+data_obj.label+"/"+folder_list[2]

    data_obj.cv_corpus = put_in_string(cv_path)
    data_obj.test_corpus = put_in_string(test_path)
    data_obj.train_corpus = put_in_string(training_path)


# need to fix this, need to make data.corpuses and data.test set
def setup():
    data = data_collection()
    data.root = "data_partitioned/"
    data.styles = {0: 'formal', 1: 'informal', 2: 'legal', 3: 'old'}

    # categories for each set of data
    data.categories = {"enron": 0, "inaugural": 0, "nps": 1, "webtext": 1, "brown_govt": 2, "supreme_corpus": 2, "gutenberg": 3}
    data.cats = []

    data.sources = []

    folders = ["cv_data/", "test_data/", "training_data/"]

    # create a new data set object for each key
    # open all the files
    # put training set stuff into training corpus string,
    # put test stuff into test corpus string, put cv into cv corpus string
    for key in data.categories:
        # print key
        new_data = data_sets()
        new_data.label = key
        new_data.style_code = data.categories[key]
        new_data.style = data.styles[new_data.style_code]
        initialize_data(new_data, data.root, folders)
        data.sources.append(new_data)
        data.cats.append(new_data.style_code)
        # print new_data.style_code

    data.train_data = []
    for data_source in data.sources:
        data.train_data.append(data_source.train_corpus)

    return data


def check_setup():
    data = setup()
    for obj in data.sources:
        print "data label is: "+obj.label
        print "data style is: "+obj.style
        train_corpus = obj.train_corpus
        cv_corpus = obj.cv_corpus
        test_corpus = obj.test_corpus
        print "length of training corpus is: " + str(len(train_corpus))
        print "length of cv corpus is: " + str(len(cv_corpus))
        print "length of test corpus is: " + str(len(test_corpus))
        # print "First 100 characters of training are: "+train_corpus[:100]
        print "\n"

# check_setup()