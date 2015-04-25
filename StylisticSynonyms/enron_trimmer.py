__author__ = 'Shashank'
import os, sys


def partition(string):
    s = string.split('X-FileName: ')
    str = s[1].split("\r")
    i = 1
    m = ""
    while i < len(str):
        m += str[i] + "\r"
        i += 1
    return m

counter = 1
for folder, sub_folders, documents in os.walk('enron/'):
    # print sub_folders
    # print folder
    if "all_documents" in folder:
        print "here"
        for each_file in documents:
            print each_file
            text_file = open("data/enron/document_"+str(counter), "w")
            new_str = open(folder+"/"+each_file, "r").read()
            x = partition(new_str)
            text_file.write(x.encode('utf-8'))
            counter += 1
