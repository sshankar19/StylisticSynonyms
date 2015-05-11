__author__ = 'Shashank'
import requests
from nltk.corpus import wordnet as wn

def find_wn_syns(word, pos):
    for syn_set in wn.synsets(word, pos):
        syns = [n.replace('_', ' ') for n in syn_set.lemma_names()]
        # print '  synonyms:', ', '.join(syns)
        return syn_set.lemma_names()


def find_syns_big_huge(word, pos):
    req = requests.get(big_url+big_api_key+word+"/json")
    j_son = req.json()
    # print type(j_son)
    # print j_son
    values = j_son[pos]
    # print pos
    # print values
    if "syn" in values:
        return values["syn"]
    return []


def merge(list1, list2):
    return list1 + list(set(list2)-set(list1))


def find_syns_wnik(word):
    req = requests.get(wnik_url_1+word+wnik_url_2+wnik_api_key)
    j_son = req.json()
    # print type(j_son)
    synonyms = []
    equivalent = []
    for key in j_son:
        if "synonym" in key.values():
            synonyms = key["words"]
        elif "equivalent" in key.values():
            equivalent = key["words"]
    # print synonyms
    # print equivalent
    return merge(synonyms, equivalent)
    # else:
    #     synonyms = []
    # if "equivalent" in j_son:



def switch_big_huge(x):
    pos_converter = {
        wn.ADJ: "adjective",
        wn.NOUN: "noun",
        wn.VERB: "verb",
        wn.ADV: "adverb",
    }
    return pos_converter.get(x, None)


def switch_wn(x):
    pos_converter = {
        "JJ": wn.ADJ,
        "JJR": wn.ADJ,
        "JJS": wn.ADJ,
        "MD": wn.VERB,
        "NN": wn.NOUN,
        "NNS": wn.NOUN,
        "NNP": wn.NOUN,
        "NNPS": wn.NOUN,
        "PRP": wn.NOUN,
        "PRP$": wn.NOUN,
        "RB": wn.ADV,
        "RBR": wn.ADV,
        "RBS": wn.ADV,
        "VB": wn.VERB,
        "VBD": wn.VERB,
        "VBG": wn.VERB,
        "VBN": wn.VERB,
        "VBP": wn.VERB,
        "VBZ": wn.VERB,
        "WP": wn.NOUN,
        "WRB": wn.ADV,
    }
    return pos_converter.get(x, None)


wnik_url_1 = "http://api.wordnik.com:80/v4/word.json/"
wnik_url_2 = "/relatedWords?useCanonical=true&api_key="

big_url = "http://words.bighugelabs.com/api/2/"

# print find_syns_wnik("exact", 2)


def find_syns(word, tag):
    wn_pos = switch_wn(tag)
    # print wn_pos
    big_pos = switch_big_huge(wn_pos)
    # print big_pos
    wn_syns = find_wn_syns(word, wn_pos)
    # print "wn_syns"
    # print wn_syns
    big_syns = find_syns_big_huge(word, big_pos)
    # print "big_syns"
    # print big_syns
    wnik_syns = find_syns_wnik(word)
    # print "wnik_syns"
    # print wnik_syns
    first = merge(wn_syns, big_syns)
    second = merge(first, wnik_syns)
    return second

# syns = find_syns_big_huge("greater", "adjective")
