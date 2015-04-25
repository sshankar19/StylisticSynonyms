__author__ = 'Shashank'

from nltk.corpus import inaugural, gutenberg, webtext, brown, nps_chat
import os, re


def createdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

root = createdir("data/")


def createdir2(directory):
    createdir(root+directory)
    return root+directory


def write_to_file(path, new_file, corpus):
    try:
        with open(path+new_file) as f:
            pass
    except IOError:
        text_file = open(path+new_file, "w")
        x = corpus.raw(new_file)
        text_file.write(x.encode('utf-8'))
        text_file.close()


def write_to_file_remove_tags(path, new_file, corpus):
    try:
        with open(path+new_file) as f:
            pass
    except IOError:
        text_file = open(path+new_file, "w")
        x = corpus.raw(new_file)
        x = re.sub(r'/([\w]*[-\ww]*)|(\-hl)', ' ', x)
        text_file.write(x.encode('utf-8'))
        text_file.close()


def write_post_list(path, posts):
    new_path = "NPS_Chats"
    try:
        with open(path+new_path) as file:
            pass
    except IOError:
        text_file = open(path+new_path, "w")
        for post in posts:
            text_file.write(post.text.encode('utf-8'))
            text_file.write("\n")
        text_file.close()


easy_files = {
    inaugural: createdir2("inaugural/"),
    gutenberg: createdir2("gutenberg/"),
    webtext: createdir2("webtext/"),
    }

for key in easy_files:
    for addr in key.fileids():
        write_to_file(easy_files[key], addr, key)

brown_govt_root = createdir2("brown_govt/")
govt_files = brown.fileids(categories="government")

for govt in govt_files:
    write_to_file_remove_tags(brown_govt_root, govt, brown)

nps_chat_root = createdir2("nps/")
posts = nps_chat.xml_posts()
post_list = []
for p in posts:
    if p.attrib['class'] != "System":
        post_list.append(p)
write_post_list(nps_chat_root, post_list)

shakespeare_root = createdir2("shakespeare/")

