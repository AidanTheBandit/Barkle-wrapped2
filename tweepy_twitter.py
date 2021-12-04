import tweepy
import re
import textblob
#from textblob import TextBlob
import twitter_credentials


# Client
def getClient():
    client = tweepy.Client(bearer_token = twitter_credentials.bearer_token,
                           consumer_key = twitter_credentials.consumer_key,
                           consumer_secret = twitter_credentials.consumer_secret,
                           access_token = None, access_token_secret = None)
    return client


# User info
def getUserInfo(user):
    client = getClient()
    user = client.get_user(username=user)
    return user.data


# Tweets
def getUserRecentTweets(id):
    client = getClient()
    user_tweets = client.get_users_tweets(id = id,
                                          tweet_fields = ['public_metrics'],
                                          exclude = ['retweets','replies'],
                                          max_results = 100,
                                          #start_time = '2021-09-02T00:00:00.000Z'
                                          )
    return user_tweets


# Get user recent tweets
def storeUserTweets(username):
    user = getUserInfo(username) # get user info, such as id
    user_tweets = getUserRecentTweets(user.id) # get tweets of user by id
    
    if len(user_tweets.data) > 0:
        for x in user_tweets.data:
            print(clean_tweet(str(x)))
    else:
        print("No tweets")
            
            
# credit freeCodeCamp.org
# removes special characters and hyperlinks
# \' escapes apostrophes
def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z’' \t])|(\w+:\/\/\S+)", " ", tweet).split())

storeUserTweets('tweetwrapped')