from tqdm import tqdm
import pickle
import time
import pandas as pd
import sys, os, csv
from tweetokenize import *
from collections import defaultdict
import re
import string
import nltk

import langdetect, langid, fasttext, cld3

from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk import bigrams
from urllib.parse import urlparse

tqdm.pandas()

curdir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

regex = re.compile('[^a-zA-Z ]')
stemmer = LancasterStemmer()
stopwords = set(stopwords.words('english'))
custom_stopwords = [line.strip() for line in open(os.path.join(curdir, 'stopwords-en.txt'))]
stopwords.update(set(custom_stopwords))


#model downloaded from: https://fasttext.cc/docs/en/language-identification.html 
fasttext_lang_detect_model = fasttext.load_model(os.path.join(curdir, 'models', 'fasttext-lang-detect-model.bin'))

def printf(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def preprocess_text(document, stem=False):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(document))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        tokens = [word for word in document.lower().split() if word not in stopwords and len(word)>2]

        if stem:
            tokens = [stemmer.lemmatize(word) for word in tokens]

        return tokens


# def remove_url(text, replacement=''):
#     return re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', replacement, text)


def preprocess_tweet(tweet, pos=False,  stem = False):

    #tokenizer will remove the parameters
    tokenizer = Tokenizer(usernames="USERNAME", urls='URL', hashtags=False, phonenumbers='', 
                times='', numbers='', ignorequotes=True, ignorestopwords=False) 


    text= tweet.text
    if 'urls' in tweet and tweet.urls: #replace short urls with expanded version
        for url in tweet.urls:
            text = text.replace(url['url'], urlparse(url['expanded_url']).netloc.replace('.',' url '))

    # Removing prefixed 'b'
    text = re.sub(r'^b\s+', '', text)

    # tokenize the texts
    tokens = tokenizer.tokenize(text)
    
    # remove stop words and words less than 3 characters
    tokens = [token.lower() for token in tokens if len(token.strip())>2] 


    if pos: # calculate POS
        tokens = nltk.pos_tag(tokens)

    if stem:
        try:
            tokens =  [(stemmer.stem(t[0]), t[1]) for t in tokens if len(stemmer.stem(t[0]))>1]
        except Exception as e:
            print(e, tokens)
#
    return tokens

def stem(words):
    return [stemmer.stem(w) for w in words]

def jaccard(s1, s2):
    #print('calculating jaccard {},{}'.format(s1,s2))
    u = len(s1.union(s2))
    i = len(s1.intersection(s2))
    #print('u:',u)
    #print('i:',i)
    if u == 0:
        return 0
    return i*1./u

def remove_entities(tweet):
    text = tweet.text
    for u in tweet.urls:
        text = text.replace(u['url'],'')
    for t in tweet.tags:
        text = text.replace('#'+t['text'], '')
    for m in tweet.mentions:
        text = text.replace('@'+m['screen_name'], '')
    return text
        
def important_words(text):
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
    return [word for word in text.lower().split() if word not in stopwords and len(word)>=3]

def clean_text(text):
    text = re.split(r'[,;.?!@#$%^&*()~\'\" ]+', text.lower())
    text = [t for t in text if len(t)>=3 and t not in stopwords]
    return text

def remove_html_tags(text):
    return text.replace('&nbsp','').replace('&amp','')
    
def text_similarity(text1, text2):
    return jaccard(set(important_words(text1)), set(important_words(text2)))
    
def tweet_similarity(tweet1, tweet2):
    return text_similarity(remove_entities(tweet1), remove_entities(tweet2))
    
#calculate tag similarity of tweets t1 and t2 
def tag_similarity(t1, t2):
    tags1 = set(tag['text'].lower() for tag in t1.tags)
    tags2 = set(tag['text'].lower() for tag in t2.tags)
    return jaccard(tags1, tags2)

def tag_reuse(user_df, tweet_df, th=50, interval=3):
    similarities = []
    for uid in user_df.index.get_level_values(0):
        tweets = tweet_df.loc[uid]
        viral_tweet_ids = tweets[tweets.retweets>=th].tid.values
        pre = 0
        post = 0
        for tid in viral_tweet_ids:
            v_tweet = tweets.loc[tid]
            v_date = v_tweet.created_at
            post_tweets = list(tweets[tweets.created_at>= v_date][
                tweets.created_at<= v_date+ datetime.timedelta(days=interval)])
            print((post_tweets[0]))
            similarity = np.mean([tag_similarity(v_tweet, t) for t in post_tweets])
        similarities.append(similarity)
    return similarities
        
def most_freq_words(tweets, n = 10, percent = False):
    word_count = defaultdict(int)
    for t in tweets:
        t = regex.sub('',t)
        words = [word for word in t.lower().split() if word not in stopwords and len(word)>=3]
        for w in words:
            word_count[w]+= 1
    n = int(len(word_count) * n / 100) if percent else n
    return [(k, word_count[k]) for k in sorted(word_count, key=word_count.get, reverse=True)][:n]
    
def most_freq_tags(all_tags, n = 10, percent = False):
    tag_dict = defaultdict(int)
    for tags in all_tags:
        for tag in tags:
            tag_dict[tag['text'].lower()] += 1
    n = int(len(tag_dict) * n / 100) if percent else n
    return [(k, tag_dict[k]) for k in sorted(tag_dict, key=tag_dicttag_count.get, reverse=True)][:n]

def starts_with_ques(text):
    ques = [
            'who', 'what', 'where', 'why', 'how', 'when',
            'does','do', 'did', 'was', 'were', 'is', 'are', 'can','could','should','would','will','shall'
          ]
    token = text.split()[0]
    
    for q in ques:
        if text.startswith(q):
            return True
    return False

def is_question(tweet):
    text = remove_entities(tweet)
    #print(text)
    #print( '?' in text)
    #print(starts_with_ques(text))
    if '?' in text and starts_with_ques(text) :
        return True
    return False
    
def n_word_block(word_list, n=2):
    '''
    Given a list of words, generates a list of
    n-word blocks
    '''
    for i in range(0, len(word_list) - n + 1):
        yield word_list[i:i+n]

def create_preprocessed_tweet_data(data_frame_file, outfile, append=False):
    '''
    Preprocess tweets and write preprocessed text from a tweet in a line
    '''
    
    print('Loading dataframe file: {} ...'.format(data_frame_file), end='', flush=True)
    tweet_df = pd.read_pickle(data_frame_file)
    print('done.\nCleaning text ...')
    tweet_df['clean_text'] = tweet_df.progress_apply(lambda tweet: ' '.join(preprocess_tweet(tweet, pos=False)), axis=1)
    print('done.\nSaving files...')
    tweet_df.to_pickle(data_frame_file)
    
    with open(outfile,'a' if append else 'w') as file:
        tweet_df.progress_apply(lambda tweet: file.write(tweet.clean_text+'\n'), axis=1)
    
    print('\n all done.')
    


def detect_lang(text, detector='fasttext'):
    try:
        if detector=='langdetect':
            return langdetect.detect(text)
        if detector=='fasttext':
            return fasttext_lang_detect_model.predict(text)[0][0][-2:]
        if detector=='langid':
            return langid.classify(text)[0]
        if detector=='cld3':
            return  cld3.get_language(text).language
    except Exception as e:
        print('Text: {} \ndetector: {} \nerror:{}'.format(text, detector, e))
        return 'NA'

