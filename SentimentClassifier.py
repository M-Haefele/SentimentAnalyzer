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

#create a list of users(maybe ~20, and have the classifer run on each user)
twitterUsers = [ 
    '@nprpolitics', 
    '@donnabrazile',
    '@politifact',
    '@AOC',
    '@mtgreenee', 
    '@SenSanders', 
    '@GovRonDeSantis', 
    '@CNN',
    '@BBC',
    '@FoxNews']

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
        df.to_csv(r'~/Desktop/BigDataProj/BigDataProj/TestCorpuses/'+user+'.csv', index = False)
    
def trainModel():
    parseTrainingCSV()
    trainingSetPreprocessor(negativeTraining, "negative") 
    trainingSetPreprocessor(positiveTraining, "positive")
    #trainingSetPreprocessor(neutralTraining, "neutral")
    userTweets = parseTestCSV()
    classifier(userTweets)
    #testSetPreprocessor()

def parseTrainingCSV():
    #opening the stanford CSV file
    with open('stanford.csv', mode ='r',encoding="utf-8")as file:
        csvFile = csv.reader(file)
        negative = 0
        positive = 0
        for lines in csvFile:
            try:
                #need to import this correctly
                if lines[0] == "0" and negative < 100:
                    negativeTraining.append(lines[5])
                    negative = negative + 1
                if lines[0] == "2":
                    neutralTraining.append(lines[5])
                if lines[0] == "4" and positive < 100:
                    positiveTraining.append(lines[5])
                    positive = positive + 1
            except Exception as e: print(e)


def parseTestCSV():
    userTweets = []
    entries = os.listdir('/home/matt/Desktop/BigDataProj/BigDataProj/TestCorpuses')
    for tweets in entries:
        with open('/home/matt/Desktop/BigDataProj/BigDataProj/TestCorpuses/'+tweets, mode= 'r', encoding="utf-8")as file:
            csvFile = csv.reader(file)
            for lines in csvFile:
                try:
                    userTweets.append((lines[0], tweets))
                except Exception as e: print(e)
    return userTweets

def clean(temp):
    stop_free = " ".join([i for i in temp.lower().split() if i not in stop if i not in negative])
    punc_free = "".join([ch for ch in stop_free if ch not in exclude])
    normalized = " ".join([lemma.lemmatize(word) for word in punc_free.split()])
    return normalized


def trainingSetPreprocessor(corpus, sentiment):
    global sanitizedNegative
    global sanitizedPositive
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

#def testSetPreprocessor(corpus):
    #todo sanitize test data before classification
 #   pattern = re.compile(',')
  #  space_re = re.compile(r'\s+')
   # for tweet in corpus:
    #     temp_actual = TextBlob(tweet.replace("\n" , " "))
     #    temp = clean(temp_actual)
    






def classifier(userTweets):
     global sanitizedPositive
     global sanitizedNegative
     global sanitizedTest
     cl = NaiveBayesClassifier(sanitized)
     for tweet in userTweets:
         user = tweet[1]
         comment = tweet[0]
         print("\nTweet by " + user + ": " + comment + "\n" + "classified as: " +  cl.classify(comment))


        
def visualizer():
     #tweet averages from test set
     df = pd.read_csv('~/Desktop/BigDataProj/BigDataProj/Pokemon.csv')
     #tweet averages from test set
     #format of this data frame is:
     # df.read(absolutepath to CSV of classified users.csv) with format of each entry as : [ 'user', 'subjectivity', 'sentiment'] ...
     #df = pd.DataFrame(data, colums = "Twitter User", "Subjectivity", "Sentiment") 
     #clustering options for tweet quality
     types = df['Type 1'].isin(['Grass', 'Fire', 'Water'])
     #might not apply here since we've already pre-parsed tweets into small DF and CSV
     drop_cols = ['Type 1', 'Type 2', 'Generation', 'Legendary', '#']
     df = df[types].drop(columns = drop_cols)
     print(df.head())
     #trivial k means, template for tweet validity scatter
     kmeans = KMeans(n_clusters=3, random_state =0)
     #multivariate clustering, probbaly (sentiment, subjectivity)
     df['cluster'] = kmeans.fit_predict(df[['Defense', 'Attack']])
     #centroid generation 
     centroids = kmeans.cluster_centers_
     #centroid for validity
     cen_x = [i[0] for i in centroids]
     #centroid for distance?
     cen_y = [i[1] for i in centroids]
     #data frame clusters accordingly
     df['cen_x'] = df.cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_y[2]})
     df['cen_y'] = df.cluster.map({0:cen_y[0], 2:cen_y[1], 2:cen_y[2]})
     #map params, each twitter users cluster can be different color and labelled as such
     colors = ['#DF2020', '#81DF20', '#2095Df']
     df['c'] = df.cluster.map({0:colors[0], 1:colors[1], 2:colors[2]})
     plt.scatter(df.Attack, df.Defense, c=df.c, alpha =0.6, s=10)
     plt.show()



if __name__ == "__main__":
    #api = connectToTweepy()
    #generateUserCorpus(twitterUsers, sys.argv[1], api)
    trainModel()
    #visualizer()                                                  
