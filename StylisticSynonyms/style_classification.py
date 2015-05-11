__author__ = 'Shashank'
from textblob import TextBlob

import classifier as clf
import get_synonyms as gs


# Step 1: load the file reader so that we can initialize the classifier
print("Starting")
classifier, data = clf.load_the_clf("best.pkl", "data.pkl")
print("Loaded the Classifier")

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

        # Step 3: Classify the style
        new_style = clf.predict_new([style], classifier, data)

        # Step 4: Find the synonyms of the word to find (WordNet)
        syns = gs.find_syns(word_to_find, tag)

        # Step 5: Append all the synonyms to the style and classify each one by one. Finding the probabilities
        # (check against the style)
        syn_probabilities = {}
        for synonym in syns:
            # print style+synonym
            syn_probabilities[synonym] = clf.check_against_pipe([style+" "+synonym], classifier, data, new_style)

        # Step 6: Choose the highest probabilities for the correct style

        max_prob = max(syn_probabilities, key=syn_probabilities.get)

        print("Best synonym for the style is: " + max_prob)

clf.save_pickled_classifier("best.pkl", classifier)
clf.save_the_data("data.pkl", data)