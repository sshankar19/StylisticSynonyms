__author__ = 'Shashank'

import os


class data_sets(object):
    pass


class data_collection(object):
    pass


# def convert_to_pos(the_string):
#     text = word_tokenize(the_string)
#     tuple_list = pos_tag(text)
#     my_list = [x[1] for x in tuple_list]
#     new_string = " ".join(str(x) for x in my_list)
#     return new_string


def put_in_string(path):
    string = ""
    for folder, sub_folders, documents in os.walk(path):
        for each_file in documents:
            k = open(path+each_file)
            for line in k:
                string += line
            k.close()
    return string


def new_arrangement(path):
    string_list = []
    for folder, sub_folders, documents in os.walk(path):
        for each_file in documents:
            k = open(path+each_file)
            string = ""
            for line in k:
                string += line
            k.close()
            string_list.append(string)
    return string_list


def split_string(line):
    n = len(line)/100
    return [line[i:i+n] for i in range(0, len(line), n)]


# change references to put in string to new_arrangement
def initialize_data(data_obj, root, folder_list):
    cv_path = root+data_obj.label+"/"+folder_list[0]
    test_path = root+data_obj.label+"/"+folder_list[1]
    training_path = root+data_obj.label+"/"+folder_list[2]

    list_cvs = split_string(put_in_string(cv_path))
    list_tests = split_string(put_in_string(test_path))
    train_corp = put_in_string(training_path)
    data_obj.cv_corpus = list_cvs  # put_in_string(cv_path)
    data_obj.test_corpus = list_tests  # put_in_string(test_path)
    data_obj.train_corpus = train_corp

    # # new features, part of string tagging
    # data_obj.train_pos = convert_to_pos(train_corp)
    # data_obj.cv_pos = [convert_to_pos(x) for x in list_cvs]
    # data_obj.test_pos = [convert_to_pos(x) for x in list_tests]


#
# def initialize_data(data_obj, root, folder_list):
#     cv_path = root+data_obj.label+"/"+folder_list[0]
#     test_path = root+data_obj.label+"/"+folder_list[1]
#     training_path = root+data_obj.label+"/"+folder_list[2]
#
#     data_obj.cv_corpus = new_arrangement(cv_path)
#     data_obj.test_corpus = new_arrangement(test_path)
#     data_obj.train_corpus = new_arrangement(training_path)

# need to fix this, need to make data.corpuses and data.test set


def setup():
    data = data_collection()
    data.root = "data_partitioned/"
    data.styles = {0: 'formal', 1: 'informal', 2: 'legal', 3: 'old'}

    # categories for each set of data
    data.categories = {"enron": 0, "inaugural": 0, "nps": 1, "webtext": 1, "brown_govt": 2, "supreme_corpus": 2, "gutenberg": 3, "old_byu_text": 3}
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

    # data.train_data = []
    # for data_source in data.sources:
    #     for corpus in data_source.train_corpus:
    #         # print corpus
    #         data.train_data.append(corpus)
    #         data.cats.append(data_source.style_code)
    # return data

    data.train_data = []
    for data_source in data.sources:
        data.train_data.append(data_source.train_corpus)

    return data


# def new_check_setup():
#     data = setup()
#     for obj in data.sources:
#         print "data label is: "+obj.label
#         print "data style is: "+obj.style
#         train_corpus = obj.train_corpus
#         cv_corpus = obj.cv_corpus
#         test_corpus = obj.test_corpus
#         print "length of training corpus is: " + str(len(train_corpus))
#         print "length of cv corpus is: " + str(len(cv_corpus))
#         print "length of test corpus is: " + str(len(test_corpus))
#         print "First 100 characters of training are: "+train_corpus[:100]
#         print "\n"


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