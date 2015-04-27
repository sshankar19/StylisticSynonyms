__author__ = 'Shashank'
from nltk.corpus import wordnet as wn
from textblob import TextBlob


def find_syns(word, pos):
    for syn_set in wn.synsets(word, pos):
        syns = [n.replace('_', ' ') for n in syn_set.lemma_names()]
        print '  synonyms:', ', '.join(syns)
        return syn_set.lemma_names()


def switch(x):
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

# Step 1: load the file reader so that we can initialize the classifier

# (vectorizer, tmatrix, classifier, data) = fr.classify(fr.setup())


# Step 2: Ask for raw_input over and over:
while raw_input("Do you want to find synonyms?").lower() != "no":

    style = raw_input("Enter your document thus far: ")
    word_to_find = raw_input("Enter the word that you want to find a synonym for: ")
    word_usage = raw_input("Enter a small sentence demonstrating usage of the word ")

    # Step 2a: Find the part of speech (with TextBlob)

    sentence_blob = TextBlob(word_usage)
    tag = None
    for tag_tuple in sentence_blob.tags:
        if tag_tuple[0].lower() == word_to_find.lower():
            tag = tag_tuple[1]

    if tag is None:
        print "Sorry the sentence doesn't have the word in it"
    else:
        part_of_speech = switch(tag)

    print part_of_speech
    # Step 3: Classify the style

    # new_style = fr.predict_new(style, classifier, data)

    # Step 4: Find the synonyms of the word to find (WordNet)

    syns = find_syns(word_to_find, part_of_speech)

    # Step 5: Append all the synonyms to the style and classify each one by one. Finding the probabilities
    # (check against the style)
    # print syns
    # syn_styles = {}
    syn_probabilities = {}

    # for synonym in syns:
        # syn_probabilities[synonym] = fr.get_prob((style+synonym), data, classifier, new_style)

    # Step 6: Choose the highest probabilities for the correct style

    max_prob = max(syn_probabilities, key=syn_probabilities.get)

    print("Best synonym for the style is: " + max_prob)