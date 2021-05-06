#API imports
import tweepy
import config
import webbrowser

#System imports
import sys 
import getopt
import time
import re
import csv
import os

#NLP, Bayes and Data visualization imports
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
import string
from textblob.classifiers import NaiveBayesClassifier
import scipy.stats as st

#create a list of users(maybe ~20, and have the classifer run on each user)
twitterUsers = [
    '@BarackObama',
    '@justinbieber',
     '@katyperry', 
    '@donnabrazile',
    '@politifact',
    '@AOC',
    '@mtgreenee', 
    '@SenSanders', 
    '@GovRonDeSantis', 
    '@CNN',
    '@BBC',
    '@FoxNews'
]

#global parameters for model

output = []
usersList = []
negativeTraining = []
positiveTraining = []
neutralTraining = []
sanitized = []
sanitizedTest = []
stop = set(stopwords.words('english')) 
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
classifiedUsers = []

negative = ['not', 'neither', 'nor', 'but', 'however', 'although', 'nonetheless', 'despite', 'except',
                         'even though', 'yet']


def connectToTweepy():
    callback_uri = 'oob' #callback
    auth = tweepy.OAuthHandler(config.consumer_key, config.consumer_secret_key, callback_uri)
    redirect_url = auth.get_authorization_url()
    webbrowser.open(redirect_url)
    user_pin_input = input("Supply pin from API : ")
    auth.get_access_token(user_pin_input)
    return tweepy.API(auth)

def generateUserCorpus(user, tweetcount, api):
    # Iterate and print tweets
    columns = set()
    allowed_types = [str, int]
    tweets = []
    for user in twitterUsers:
        timeline = api.user_timeline(user ,count=tweetcount,tweet_mode="extended")
        for status in timeline:
            #timeline parsed into dictionary
            individual_tweet = {}
            status_dictionary = dict(vars(status))
            keys = status_dictionary.keys()
            for k in keys:
                value = type(status_dictionary[k])
                if value in allowed_types:
                    individual_tweet[k] = status_dictionary[k]
                if k == 'full_text':
                    columns.add(k)
            tweets.append(individual_tweet)    
        csv_columns = list(columns)
        df = pd.DataFrame(tweets, columns=csv_columns)
        df.to_csv(r'/home/matt/SentimentAnalyzer/TestCorpuses/'+user+'.csv', index = False)
        print(df)

def trainModel():
    parseTrainingCSV()
    trainingSetPreprocessor(negativeTraining, "negative") 
    trainingSetPreprocessor(positiveTraining, "positive")
    #trainingSetPreprocessor(neutralTraining, "neutral")
    userTweets = parseTestCSV()
    testSetPreprocessor(userTweets)
    print(sanitizedTest)
    classifier(sanitizedTest)

def parseTrainingCSV():
    #opening the stanford CSV file
    with open('stanford.csv', mode ='r',encoding="utf-8")as file:
        csvFile = csv.reader(file)
        negative = 0
        positive = 0
        for lines in csvFile:
            try:
                #need to import this correctly
                if lines[0] == "0" and negative < 2000:
                    negativeTraining.append(lines[5])
                    negative = negative + 1
                if lines[0] == "2":
                    neutralTraining.append(lines[5])
                if lines[0] == "4" and positive < 2000:
                    positiveTraining.append(lines[5])
                    positive = positive + 1
            except Exception as e: print(e)


def parseTestCSV():
    userTweets = []
    entries = os.listdir('/home/matt/SentimentAnalyzer/TestCorpuses')
    for users in entries:
        with open('/home/matt/SentimentAnalyzer/TestCorpuses/'+users, mode= 'r', encoding="utf-8")as file:
            csvFile = csv.reader(file)
            for lines in csvFile:
                try:
                    userTweets.append((lines[0], users[:-4]))#truncates .csv off user name
                except Exception as e: print(e)
    return userTweets

def clean(temp):
    stop_free = " ".join([i for i in temp.lower().split() if i not in stop if i not in negative])
    punc_free = "".join([ch for ch in stop_free if ch not in exclude])
    normalized = " ".join([lemma.lemmatize(word) for word in punc_free.split()])
    return normalized


def trainingSetPreprocessor(corpus, sentiment):
    global sanitized
    pattern = re.compile(',')
    space_re = re.compile(r'\s+')
    for tweet in corpus:
        temp_actual = TextBlob(tweet.replace("\n" , " "))
        temp = clean(temp_actual)
        #words = temp.split()
        #for word in words:
        if sentiment == "negative":
            sanitized.append((temp, "negative"))
        elif sentiment == "positive":
            sanitized.append((temp, "positive"))
     #   else:
      #      sanitizedNeutral.append(temp)

def testSetPreprocessor(corpus):
    #todo sanitize test data before classification
    pattern = re.compile(',')
    space_re = re.compile(r'\s+')
    for tweet in corpus:
        temp_actual = TextBlob(tweet[0].replace("\n" , " "))
        temp = clean(temp_actual)
        sanitizedTest.append((temp, tweet[1]))   

def classifier(userTweets):
    global sanitized
    cl = NaiveBayesClassifier(sanitized)
    #each twitter corpus is its own index in larger list?
    # for user in list
    #for tweet in user
    negativeFreq = 0
    positiveFreq = 0
    tweetCount = 1
    #right now this is controlled for a user tweet sample of 100 tweets
    for tweet in userTweets:
        currentUser = tweet[1]
        comment = tweet[0]
        if cl.classify(comment) == "negative":
             negativeFreq = negativeFreq + 1
        else:
             positiveFreq = positiveFreq + 1 
        tweetCount = tweetCount + 1
        userScore = positiveFreq/ float((negativeFreq + positiveFreq))
        print(currentUser + "'s aggregate sentiment score is: " + str(userScore))
        if tweetCount == 100:
            tweetCount = 1
            positiveFreq = 0
            negativeFreq = 0 
            classifiedUsers.append((userScore, currentUser))
            print("created user cluster")
    print(classifiedUsers)
    

def visualizer():
    #tweet averages from test set
    #column_names = ['SentimentScore', 'user']
    #df = pd.DataFrame(classifiedUsers, columns=column_names)
    #savings aggregate classifier scores for prototyping etc
    #df.to_csv(r'/home/matt/SentimentAnalyzer/TestCorpuses/classifiedUsers.csv', index = False)
    df = pd.read_csv('/home/matt/SentimentAnalyzer/TestCorpuses/bbcsample.csv')
    #df = pd.DataFrame(classifiedUsers) 
    print(df)
    labels = ['Negative sentiment', 'Positive sentiment']
    x = np.arange(len(labels))
    news = [ 643, 357]
    normal_users = [565, 435]
    width = 0.5
    fig, ax = plt.subplots()
    rects1 = ax.bar(x -width/2, news, width, label= 'News and Formal Twitter')
    rects2 = ax.bar( x + width/2, normal_users, width, label = 'Average Twitter Users')
    ax.set_ylabel('Classified Frequency')
    ax.set_title('Comparative Frenquencies of Formal and Informal Twitter Users')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.show()

if __name__ == "__main__":
    #api = connectToTweepy()
    #generateUserCorpus(twitterUsers, sys.argv[1], api)
    #trainModel()
    visualizer()      
