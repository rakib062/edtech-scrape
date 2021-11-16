#!/usr/bin/env python
# coding: utf-8

# #### See the link for details on v2 Tweet object
# https://blog.twitter.com/developer/en_us/topics/tips/2020/understanding-the-new-tweet-payload
# 
# - Specify "Entity for URLS and mentions
# - "data" in the response contains tweet details
# - "include" in the response contains 
#  - user details
#  - location info
#  - whether the tweet was quoted
# - "public_metrics" nested in tweet object contains retweets, likes, etc.
# - "public_metrics" nested in user object contains followers, followes, etc.
# - "non_public_metrics" nested in tweet object contains impressions, video view, etc.
# - "context_annotations" in nested in tweet provides contextual information to help you understand what the Tweet is about without needing to do custom entity recognition.
#  - Each object within context_annotations contains a ‘domain’ object and an ‘entity’ object, and each of those have an id and name property. The domain indicates the high level category or topic under which the Tweet falls, and the entity indicates the person, place, or organization that is recognized from the Tweet text. More details on available annotations can be found on our [documentation page](https://developer.twitter.com/en/docs/labs).
#  
# See here for details of tweet object: https://developer.twitter.com/en/docs/twitter-api/data-dictionary/object-model/tweet

# In[6]:


import tweepy
from twitter_authentication import bearer_token
import pandas as pd
import datetime
import time, os, sys
import csv
client = tweepy.Client(bearer_token, wait_on_rate_limit=True)


# In[135]:


def media_to_df(response):
    '''
    Converts Media objects returned with tweets to a dataframe.
    '''
    media = []
    for m in response.includes['media']:
        media.append(
            { 
              'media_key': m.media_key,
              'media_type': m.type, 
              'alt_text': m.alt_text,
              'duration_ms': m.duration_ms
             })
    df = pd.DataFrame(media)
    df.set_index("media_key", inplace=True)
    return df

def users_to_df(response):
    '''
    Converts Users objects returned with tweets to a dataframe.
    '''
    users = []
    # Take all of the users, and put them into a dictionary of dictionaries with the info we want to keep
    for user in response.includes['users']:
        users.append(
            { 
            'userid': user.id,
            'username': user.username, 
            'followers': user.public_metrics['followers_count'],
            'tweets': user.public_metrics['tweet_count'],
            'profile_desc': user.description,
            'location': user.location,
            'verified': user.verified,
            'entities': user.entities
             })
    df = pd.DataFrame(users)
    df.set_index("userid", inplace=True)
    return df

def tweet_to_row(tweet):
    return {
        'tweetid': tweet.id,
        'author_id': tweet.author_id, 
        'text': tweet.text,
        'created_at': tweet.created_at,
        'geo': tweet.geo,
        'retweets': tweet.public_metrics['retweet_count'],
        'replies': tweet.public_metrics['reply_count'],
        'likes': tweet.public_metrics['like_count'],
        'quote_count': tweet.public_metrics['quote_count'],
        'lang':tweet.lang,
        'conversation_id': tweet.conversation_id,
        'mentions': [] if 'mentions' not in tweet.entities else tweet.entities['mentions'],
        'urls': [] if 'urls' not in tweet.entities else tweet.entities['urls'],
        'hashtags': [] if 'hashtags' not in tweet.entities else tweet.entities['hashtags'],
        'referenced_tweets': [] if 'referenced_tweets' not in tweet.entities else tweet.entities['referenced_tweets'],
        'context_annotations': tweet.context_annotations,
        'attachments': tweet.attachments,
        'possibly_sensitive': tweet.possibly_sensitive,
        'withheld' : tweet.withheld,
        'reply_settings': tweet.reply_settings,
        'source':tweet.source
        }

def included_tweets_to_df(response):
    result = []
    for tweet in response.includes['tweets']:
        result.append(tweet_to_row(tweet))

    df = pd.DataFrame(result)
    df.set_index('tweetid', inplace=True)
    return df

def tweets_to_df(response):
    result = []
    for tweet in response.data:
        result.append(tweet_to_row(tweet))

    df = pd.DataFrame(result)
    df.set_index('tweetid', inplace=True)
    return df


# In[114]:


def search_tweets(query, outdir):
    tweet_count = 0
    try:
        for response in tweepy.Paginator(
                client.search_all_tweets, 
                query = query,
                user_fields = ['username', 'public_metrics', 'description', 
                               'location', 'protected', 'verified',
                                'entities', 'url'],
                tweet_fields = ['id', 'text', 'author_id', 
                                'created_at', 'geo', 'public_metrics',
                                'lang', 'conversation_id', 'entities',
                                'referenced_tweets', 'context_annotations', 
                                'attachments', 'possibly_sensitive',
                                'withheld', 'reply_settings', 'source'
                                #'organic_metrics', #'promoted_metrics', #'non_public_metrics',
                               ],
                expansions = ['author_id', 'referenced_tweets.id', 
                              'referenced_tweets.id.author_id',
                              'in_reply_to_user_id', 'attachments.media_keys',
                              'entities.mentions.username'],
                start_time = '2006-03-21T00:00:00Z',
        #         end_time = '2021-01-21T00:00:00Z',
                place_fields=['full_name', 'id'],
                media_fields=['type', 'url', 'alt_text', 
                              'public_metrics', 'duration_ms'],
                max_results=100):

            tweet_count+=len(response.data)
            print('query: {}, tweets: {}, total: {}'.format(
                query, len(response.data), tweet_count))

            user_df = users_to_df(response)
            tweet_df = tweets_to_df(response)
            media_df = media_to_df(response)
            included_tweet_df = included_tweets_to_df(response)

            user_df.to_csv("{}/users-search-{}-{}.csv".format(
                outdir, query, datetime.datetime.now()))
            tweet_df.to_csv("{}/tweets-search-{}-{}.csv".format(
                outdir, query, datetime.datetime.now()))
            included_tweet_df.to_csv("{}/inc-tweets-search-{}-{}.csv".format(
                outdir, query, datetime.datetime.now()))
            media_df.to_csv("{}/media-search-{}-{}.csv".format(
                outdir, query, datetime.datetime.now()))

            time.sleep(5)

    except Exception as e:
        print("Exception for query:{}".format(query))
        with open('exceptions.txt', 'a') as file:
            print(query+"\n")


# In[138]:


# search_tweets(query="#edtech", outdir="tweets/")


# In[ ]:




