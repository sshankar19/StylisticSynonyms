__author__ = 'Shashank'
import os, sys
from random import shuffle
import shutil
from extractor import createdir


def set_up_dirs(path):
    createdir(path)
    print path
    data_sets = ["training_data/", "cv_data/", "test_data/"]
    for data_folder in data_sets:
        createdir(path+data_folder)
    return data_sets


def multiple_file_splitting(root):
    original_dir = "data/"+root+"/"
    new_dir = "data/split/"+root+"/"
    dirs = set_up_dirs(new_dir)

    files = []
    for folder, sub_folders, documents in os.walk(original_dir):
        for each_file in documents:
            files.append(each_file)

    shuffle(files)
    num_files = len(files)

    train_lim = 0.6*num_files
    cv_lim = 0.8*num_files

    for counter in range(num_files):
        if counter < train_lim:
            shutil.move(original_dir+files[counter], new_dir+dirs[0]+files[counter])
        elif counter < cv_lim:
            shutil.move(original_dir+files[counter], new_dir+dirs[1]+files[counter])
        else:
            shutil.move(original_dir+files[counter], new_dir+dirs[2]+files[counter])


def write_file(path, string):
    try:
        with open(path) as file:
            pass
    except IOError:
        text_file = open(path, "w")
        if type(string) is str:
            u = unicode(string, 'utf-8')
            text_file.write(u.encode('utf-8'))
        else:
            text_file.write(string.encode('utf-8'))
        text_file.close()


def file_len(fname):
    opened_file = open(fname)
    return sum(1 for line in opened_file)


def intify(x):
    return int(round(x))


def single_splitting_mk2(root, single):
    old_str = "data/"+root+single
    length = file_len(old_str)
    file_pointer = open(old_str)

    train_length = intify(length*0.6)
    cv_length = intify(length*0.8)
    print train_length
    print cv_length

    new_dir = "data/split/"+root
    dirs = set_up_dirs(new_dir)

    counter = 0
    train = ""
    cv = ""
    test = ""

    for line in file_pointer:
        if counter < train_length:
            train += line
        elif counter < cv_length:
            cv += line
        else:
            test += line
        counter += 1

    file_pointer.close()

    write_file(new_dir+dirs[0]+"__TRAIN__"+single, train)
    write_file(new_dir+dirs[1]+"__CV__"+single, cv)
    write_file(new_dir+dirs[2]+"__TEST__"+single, test)


def supreme_partition(string):
    s = string.split('+++$+++')
    # print s
    stride = ""
    counter = 0
    for line in s:
        if counter % 7 == 6:
            # print "Counter is: " + str(counter) + " Line is "+line
            stride += line
        counter += 1
    return stride


def trim_supreme_court():
    text_file = open("data/supreme_corpus/supreme.conversations.txt", "r")
    new_file = open("data/supreme_corpus/supreme_corp_trim.txt", "w")
    lines = text_file.read()
    trimmed = supreme_partition(lines)
    new_file.write(trimmed.encode('utf-8'))


# createdir("data/split")
# multiple_file_splitting('brown_govt')
# multiple_file_splitting('enron')
# multiple_file_splitting('gutenberg')
# multiple_file_splitting('inaugural')

# single_splitting_mk2("supreme_corpus/", "supreme_corp_trim.txt")
# single_splitting_mk2("supreme_corpus/", "test.txt")
# single_splitting_mk2("nps/", "NPS_Chats")
# single_splitting_mk2("webtext/", "overheard.txt")
# single_splitting_mk2("webtext/", "singles.txt")
# single_splitting_mk2("webtext/", "wine.txt")


