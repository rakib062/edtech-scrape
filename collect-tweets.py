#!/usr/bin/env python
# coding: utf-8

# In[113]:


import tweepy
from twitter_authentication import bearer_token
import pandas as pd
import datetime
import time, os, sys
import csv
client = tweepy.Client(bearer_token, wait_on_rate_limit=True)




def users_to_df(tweets):
    '''
    Converts Users objects returned with tweets to a dataframe.
    '''
    users = []
    # Loop through each response object
    for response in tweets:
        # Take all of the users, and put them into a dictionary of dictionaries with the info we want to keep
        for user in response.includes['users']:
            users.append(
                { 
                  'userid': user.id,
                  'username': user.username, 
                  'followers': user.public_metrics['followers_count'],
                  'tweets': user.public_metrics['tweet_count'],
                  'profile_desc': user.description,
                  'location': user.location
                 })
    df = pd.DataFrame(users)
    df.set_index("userid", inplace=True)
    return df

def tweets_to_df(tweets):
    result = []
    # Loop through each response object
    for response in tweets:
        # Take all of the users, and put them into a dictionary of dictionaries with the info we want to keep
        for tweet in response.data:
            # Put all of the information we want to keep in a single dictionary for each tweet
            result.append({
                           'tweetid': tweet.id,
                           'author_id': tweet.author_id, 
                           'text': tweet.text,
                           'created_at': tweet.created_at,
                           'retweets': tweet.public_metrics['retweet_count'],
                           'replies': tweet.public_metrics['reply_count'],
                           'likes': tweet.public_metrics['like_count'],
                           'quote_count': tweet.public_metrics['quote_count'],
                           'lang':tweet.lang
                          })

    df = pd.DataFrame(result)
    df.set_index('tweetid', inplace=True)
    return df


def search_tweets(query, outdir):
    tweet_count = 0
    try:
        for response in tweepy.Paginator(
                client.search_all_tweets, 
                query = query, #"COVID hoax -is:retweet lang:en",
                user_fields = ['username', 'public_metrics', 'description', 'location'],
                tweet_fields = ['created_at', 'geo', 'public_metrics', 'text', 'lang'],
                expansions = ['author_id'],
                start_time = '2006-03-21T00:00:00Z',
        #         end_time = '2021-01-21T00:00:00Z',
                max_results=500):
            
            tweet_count+=len(response.data)
            print('query: {}, tweets: {}, total: {}'.format(
                query, len(response.data), tweet_count))

            user_df = users_to_df([response])
            tweet_df = tweets_to_df([response])
            user_df.to_csv("{}/users-search-{}-{}.csv".format(
                outdir, query, datetime.datetime.now()))
            tweet_df.to_csv("{}/tweets-search-{}-{}.csv".format(
                outdir, query, datetime.datetime.now()))
            time.sleep(2)
    except Exception as e:
        print("query: {}, exception:{}".format(query, e.message))
        with open('resume-tags.csv', 'a') as file:
            file.write(','+query)


outdir = sys.argv[1]
if not os.path.exists(outdir):
        os.makedirs(outdir)
        
with open('tags.csv') as csv_file:
    reader = csv.reader(csv_file)
    tags = set(list(reader)[0])
    
donetags = set([])
if os.path.isfile('donetags.csv'): 
    with open('donetags.csv') as csv_file:
        reader = csv.reader(csv_file)
        donetags = set(list(reader)[0])

tags = tags.difference(donetags)
i=1
for tag in tags:
    print("Starting search for tag no:{} of {}, tag:{}".format(i, len(tags),tag))
    search_tweets(tag, outdir)
    donetags.add(tag)
    with open('donetags.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        writer.writerow(donetags)




