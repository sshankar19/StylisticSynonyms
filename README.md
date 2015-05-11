#Natural Language Final Project##Title: Stylistic Synonyms##Author: Shashank Shankar

Final Project for CS 533 - Natural Language taught by Professor Matthew Stone and TA Brian McMahan
## Table of Contents

###[Part 0: Introduction](#Intro)
###[Part 1: The Classifier](#Part1)
* ####[Data Styles and Sources](#Data)
* ####[Classifier & Feature Details](#Features)
* ####[Results](#results)
* ####[Conclusion and Discussion (of Classifier)](#concl)

###[Part 2: The User Interface and Synonyms](#Part2)
* ####[Overview](#overview)
* ####[How To Use](#usage)
* ####[Conclusion (for Part 2)](#part2concl)

###[Part 3: References](#part3)
##[Part 0: Introduction](id:Intro)

The purpose of my Natural Language final project was to make something that both utilized what I have learned in the class and also something I found interesting. Simply put, I had to find a project that I *wanted to do*, something that would be *beneficial to me*. At the same time as  I was deciding on an idea for the project, I was also applying to different companies for a job post-graduation. 

A tedious process, I hated everything about applying to jobs, including filling out the application, interviewing, updating my resume, etc. One particular item I disliked was writing cover letters and working on my resume, especially writing formally. Since I am not particularly gifted at vocabulary, I spent a lot of time looking up synonyms for words to use. These were synonyms sounded more formal or 'smart' than the original words I had thought of. At that point *it hit me*. What if I could make a program that decided synonyms for me, that went through and chose the one that sounded the smartest or was perfect given the context?

So I made a program which suggested synonyms for a user depending on the style that they were writing in.

> What if I could make a program that decided synonyms for me, that went through and chose the one that sounded the smartest or was perfect given the context?

My program consists of two different parts: 

0. Stylistic Classifier (**classifier.py**)- A Multinomial Naive bayes Classifier that classifies text passages based on the style of text it most closely resembles
1. Synonym Finder (**style_classification.py** and **get_synonyms.py**)- Where the program user inputs: 
	* His current passage of text - (the program analyzes it to find the best style)
	* and the word for which the user wants to find the right synonyms - (the program picks the best synonoyms based on that style)


## [Part 1: The Classifier](id:Part1)
### [Data Styles and Sources](id:Data)

I decided four different styles of text at the beginning:

* Formal Writing - Characterized by more complex language, with more a diverse and distinct vocabulary as well as sentence variation. Usually in third person and objective in nature, defined by a sense of professionalism. Some examples of what formal writing is used for includes Corporate Emails, Cover Letters, Research Papers, etc.
* Informal Writing - Characterized by colloqualisms, contractions, slang, maybe even creole. This is usually used in correspondence with friends, family or for purposes where your writing does not especially need credence. Examples include instant message chats, texts, forums, Yelp reviews, etc.
* Legal Government Writing - This includes writing like the Constitution or court cases or laws and bills. I was curious to see the overlap between this style of writing and formal writing as they both seem similar to each other
* Old English writing - This category was primarily for fun, to see how different Old English was from the other styles. *Old English* may not be the best way to describe this category, as it includes texts from a wide range of dates, including from Geoffrey Chaucer to William Shakespeare to writings from the 1800s. Since English has invariably changed over that span of time, this style could use more refining


The Data Sources I used include:

*Figure 1.*

| Style | Data title | Data Source |
| -----------------------------|
| Formal | Enron Email corpus | Courtesy of William W. Cohen, MLD, CMU, <https://www.cs.cmu.edu/~./enron/> |
| Formal   | Presidential Inaugural Addresses  |  One of the corpuses from NLTK package |
| Informal | NPS Chat corpus |  One of the corpuses from NLTK package |
| Informal	| Webtext Corpus |  One of the corpuses from NLTK package |
| Legal		| Brown Government Corpus |  One of the corpuses from NLTK package |
| Legal		| Supreme Court Decision Corpus |  One of the corpuses from NLTK package |
| Old English | Project Gutenberg Corpus | One of the corpuses from NLTK package |
| Old English | BYU THE CORPUS OF HISTORICAL AMERICAN ENGLISH (COHA) |  Davies, Mark. (2010-) The Corpus of Historical American English: 400 million words, 1810-2009. Available online at <http://corpus.byu.edu/coha/>. | 

The Following Python files were used to trim and *munge* or *wrangle* the data:

* **extractor.py** 
* **assorter.py** - This was used specifically to split my data set into 60% training, 20% cross_validation, and 20% test
* **enron_trimmer.py**


### [Classifier & Feature Details](id:Features)

#### Classifier

The classifier I decided on using (since this is a text classification system) was a Multinomial Naive Bayes classifier. Originally I was deciding between this and a K-NN classifier but chose the former because prediction would be faster. I did not have enough time to use Linear SVMs but I would like to extend this to try those in the future.

For reference, I used scikit-learn's Multinomial Naive Bayes Classifier, as described [here](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB "MultinomialNB").

I also created my own classifier using scikit learn to implement additional features, extending scikit-learn's [BaseEstimator](http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html) and [ClassifierMixin](http://scikit-learn.org/stable/modules/generated/sklearn.base.ClassifierMixin.html). However I had trouble using it with my project, as described later. My stable classifier consists only of the Naive Bayes Classifier.

#### Features

I contemplated a lot of features for help with classification, these included:

* **unigrams up until 5-grams** - Unigrams as a feature was my baseline classifier, however I experimented with ngram ranges from 1 to 5 grams. I wanted to see if similar style texts had similar patterns in word sequences
* **average lengths of sentences** - Different styles of texts would have different lengths of sentences, I thought that formal and legal styles of writing would have longer sentences on average over the classes, and was interested in seeing how it would affect the results
* **stop-words** - Using stop words would greatly help in trimming out words that are overly common to each style. This would help the classifier determine the unique words per style
* **stemming** - Stemming according to Wikipedia, is the "process for reducing inflected (or sometimes derived) words to their word stem, base or root for". I used NLTK's Porter Stemmer for this purpose, but I used code provided by [Duke University](http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html) in implementing this feature
* **idf vs. non-idf** - Using IDF or inverse document frequency I assumed would help find the unique words for styles whereas non-IDF would help for finding commonalities through brute force
* **term frequency vs. log term frequency (sublinear)** - Using a sublinear term frequency may help with scaling, I didn't want to over fit for specific terms
* **sentence length variation and sentence word length** - I had difficulty implementing these couple of features, but they were both ideas for helping to differentiate formal and legal english styles from informal. I expected both of those styles to have greater variations in the sentences and a more complex vocabulary than informal
* **part-of-speech tagging** - The last feature is the result of [this paper](http://research.microsoft.com/pubs/68964/coling2004_authorship.pdf) by Michael Gamon from Microsoft Research. I did not have any predictions on how this would affect the classifier.

Average Sentence Length, Sentence Length Variability, Average Word Length, Stemmer, and Part of Speech tagging, as well as my classifier were implemented in **alt_features.py**


### [Results](id:results)

The results from the features are below, I used my method **cv_test_pipe()** in my **classifier.py** to compare the features


*Figure 2.*

| Features (with base unigram, with regular TF and using IDF) | Success rate |
| ------------ | ------------- | ------------ |
| N-gram clf (1 to 5 grams) with Stop Words and Stemming | 0.95297029703 |
| N-gram (1 to 3 grams) with Stop Words and Stemming | 0.946782178218 | 
| N-gram (1 to 3 grams) with Stop Words, Stemming and Use IDF = False | 0.938118811881 |
| Base with Stop Words and Stemming | 0.936881188119 |
| Stop Words | 0.913366336634 |
| Stemming | 0.872524752475 |
| Use-IDF = False | 0.818069306931 |  
| ***Base Unigram*** | 0.759900990099   |
| N-grams (1 to 3 grams)  | 0.75495049505  | 
| Sublinear-TF = True | 0.502475247525 | 


Out of this set, the top set of features seems to be the **N-gram clf (1 to 5 grams) with Stop Words and Stemming**, however that classifier was too large and took too long to use, so I opted to use #2 in that list **N-gram (1 to 3 grams) with Stop Words and Stemming** as my candidate as it was much smaller in size, only 4.7 GB when pickled


#### Unsuccessfully implemented Features

It may be noticed that I had some features that were not shown in that chart, namely the Sentence Length features, the Average Word Length features and the Part of Speech tagging features. There results are seen here:

*Figure 3.*

| Features | Success rate |
|----------|--------------|
| Average Sentence Length | 0.474 |
| Sentence Variation | 0.25 |
| Average Word Length | 0.25 | 
| Part of Speech Tagging | Inconclusive |

The problems with these features (or at least their implementation) are as follows:

* Sentence Variation and Average Word Length - These two features may be buggy in their implementation, which upon inspection does not make sense, because the feature matrixes produced by both seem accurate. A 0.25 score means that the classifier basically randomly classified
* Average Sentence Length - This feature only worked with my custom classifier, but the problem wasn't the feature itself but that I could not find a way to combine it with the other features in Figure 2 that used the Naive Bayes. Other than weighing the two classifiers arbitrarily (which would just result in a worse score, than using the best classifier from Figure 2, as the disparity between the two classifier's scores was just that large) I did not know of any other way of combining the two (I think I need to be better at Python/Know scikit-learn a bit better)
* Part of Speech Tagging - I started running this feature at 2:00 PM and it was still running at 8:00 PM. I do not know if it is a problem with my feature, or if I do not have enough resources to test this on my machine. I have to run some more tests

### [Conclusion and Discussion (of Classifier)](id:concl)
#### Discussion
It is apparent that the best features to use were

* N-grams, the higher the better
* Stop words
* Stemming
* Regular Term Frequency and using IDF

These set of basic features were the most useful, although there were other features that performed well as described by Figure 2. What one can reason out of the top three features is that, by first removing all the stop words, the styles were mainly left with words unique to their styles or classes. As a result, both the n-grams feature and the stemming feature benefited greatly. Data from the same style probably uses the same unique words in the same sequence so that would result in the n-grams great improvement when used with stop words. With stemming, the unique words per style now left after removing stop words may have been more morphologically similar, making it easier to classify. Finally, it seems important to note that using IDF did not make a large difference in success rate over the lack of it. 

#### Conclusion (Or how this classifier may extend to)
There is a lot more I can try with this project, and many more things to fix. 
This includes:

* **The features that do not currently work, as described in *Unsuccessfully Implemented Features*** - I am going to try and get them to work and measure their performance as well
* **Trying out with Linear SVM** - I never tried using this classifier and I am interested in seeing how this would affect my results (might even make it better)
* **More Data** - I need to find more sources of data, to make my classifier even better, I mentioned in my presentation, the BYU Corpus of American Soap Opera, and using Tweets2011 among sources
* **Different styles** - More styles! I could try having more styles, especially of different varieties or lects. 
* **More Features** - One of the features that I did not even have a chance to look at was the use of Punctuation in different styles. There might be more I have not thought about.


## [Part 2: The User Interface and Synonyms](id:Part2) 
### [Overview](id:overview)
This part was *much* shorter to implement (and therefore less to talk about) than the first part. As soon as **style_classification.py** is run for the first time, it automatically checks if there is already a classifier and data set saved from **Part 1: The Classifier** (**Part 1** will save the two by pickling them to files). If so, congratulations! It will only take 9 minutes (approximately on my workstation) to load the two. If not, then the program will have to initialize everything, which may take some time (a long time). 

The program then asks the user if they want to look for synonyms in a for loop that runs until the user types "no" or "NO". If the user *does* want to look for synonyms, then he/she will enter what they have written until that point when prompted in the console, and will follow up by entering the word they seek and a demonstration of its usage in a sentence.

The program then uses [TextBlob](http://textblob.readthedocs.org/en/dev/) to identify the part of speech of the word to find, and searches for synonyms appropriate to the word. **get_synonyms.py** demonstrates the different methods by which synonyms are retrieved

There are three services used to get the list of synonyms:

* WordNet - This is provided by NLTK, although it is not very good at getting synonyms on its own
* [Big Huge Thesaurus](https://words.bighugelabs.com/) - That is why we also use big huge thesaurus to get even more words, and finally
* [Wordnik](https://www.wordnik.com/)

After the synonyms are retrieved, we use the classifier to classify each synonym along with the original passage and find the synonym that has the best match with the style of the passage

### [How To Use](id:usage)

With the Python environment set up correctly, using Anaconda and all of the various packages (TextBlob, WordNet, CPickle, NLTK, scipy, scikit-learn, numpy)

**Warning: My Python version is "Python 2.7.9 :: Anaconda 2.1.0 (x86_64)", I do not know if it will work with others**

**style_classification.py** is run from the command line with the following command
	
	python style_classification.py
	
After it prints the message that it has loaded the classifier and data from the .pkl files, it will print and ask the following:
	
	Do you want to find synonyms? <your_answer""

As long as the user does not submit "NO" or "no", then it will proceed to print and request the following:

	Enter your document thus far:  <your_doc_here>
	
	Enter the word that you want to find a synonym for: <your_word_here>
	
	Enter a small sentence demonstrating usage of the word: <your_sentence_here>
	
Then, after politely waiting, the user will be greeted with this message

	Best synonym for the style is:  <some_synonym>

If not, then please look through the code and why you are getting your error. If the program is unable to find your word's synonyms because of a misspelling or because you used a plural (such as harnesses, etc.) try spelling the word correctly or using the singular form.

Finally, here is an example of the program with all of the input and output:

	Starting
	Done loading data
	Loaded the Classifier
	Do you want to find synonyms?yes
	Enter your document thus far: Your receiver is currently off. Press Select to watch TV. There are several programs on the television, I change channels very
	Enter the word that you want to find a synonym for: frequently
	Enter a small sentence demonstrating usage of the word I eat frequently
	Best synonym for the style is: oftenly

### [Conclusion (for Part 2)](id:part2concl)

This part is very stable and works as anticipated. The only thing I may change is the look of the program and add a better looking user-interface rather than just the console.

## [Part 3: References](id:part3)

For references I feel obligated to list the packages used, but I'm just going to put the links to everything down here unless the website specifically asked for it. 

[Wordnik](https://www.wordnik.com/)

[Big Huge Thesaurus](https://words.bighugelabs.com/)

[TextBlob](http://textblob.readthedocs.org/en/dev/) 

[Linguistic correlates of style: authorship classification with deep linguistic 
analysis features by Michael Gamon](http://research.microsoft.com/pubs/68964/coling2004_authorship.pdf)

[Duke University, Lab 2](http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html)

William W. Cohen, MLD, CMU, <https://www.cs.cmu.edu/~./enron/> 

Davies, Mark. (2010-) The Corpus of Historical American English: 400 million words, 1810-2009. Available online at <http://corpus.byu.edu/coha/>

#####Scikit Learn links

[scikit-learn](http://scikit-learn.org/stable/)

[BaseEstimator](http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html) 

[ClassifierMixin](http://scikit-learn.org/stable/modules/generated/sklearn.base.ClassifierMixin.html)

[Multinomial Naive Bayes](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB "MultinomialNB")