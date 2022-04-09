import os, sys, pickle, re
from os.path import join
import pandas as pd
import numpy as np
import fasttext as ft
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
tqdm.pandas()
from scipy.stats import mode

sys.path.append('helpers/')
from tweetokenize import *

import text_processing, preprocess_tweet_data, config


DATA_ROOT = config.DATA_ROOT

NUM_USR_PROF_CLUSTER= '10 20 30 40 50 80 100'
ALGO_USR_PROF_CLUSTER='KMeans'


TWEET_USER_DF = join(DATA_ROOT, 'users-all.pkl') #containing profile info for all users
TWEET_USER_DF_EN = join(DATA_ROOT, 'users-en.pkl') #containing profile info for users with (detected) 'en' description
TWEET_USER_DF_EN_UNIQUE = join(DATA_ROOT, 'users-en-unique.pkl') #containing unique users with (detected) 'en' description
TWEET_USER_PROFILE_TEXT = join(DATA_ROOT, 'tweet-profile-text-en.txt') #file containing cleaned profile descriptions
CLEAN_TWEET_TEXT_EN = join(DATA_ROOT,'tweet-text-en.txt') #file containing cleaned tweet text
PROFILE_WORD_WEIGHTS = join(DATA_ROOT, 'weights.wgt') #store profile word weights
PROFILE_WORD_TFIDF = join(DATA_ROOT, 'tfidf') #store profile word tfidf scores
PROFILE_FT_MODEL = join(DATA_ROOT, 'ft-profile-sg') #fasttext model for profle desc, using skipgram to catch semantic relationships
TWEET_DF_FILE = join(DATA_ROOT, 'tweets-all.pkl')
EN_TWEET_DF_FILE = join(DATA_ROOT,'tweets-en.pkl') #dataframe with english tweets 
PRIV_TWEET_DF = join(DATA_ROOT,'tweets-priv.pkl') #dataframe containing sec/priv related tweets 
PRIV_TWEET_TEXT = join(DATA_ROOT,'tweet-text-priv.txt') #file containing sec/priv tweet texts 
PRIV_TWEET_MODEL_PATH=join(DATA_ROOT, 'priv-tweet-models/') #directory for ft and cluster models for sec/priv related tweets
PRIV_FT_MODEL  = join(PRIV_TWEET_MODEL_PATH, 'ft-priv-tweet-sg') #ft model name for sec/priv related tweets
USR_CLUSTER_DIR = join(DATA_ROOT, 'profile-cluster-models') #directory to save profile cluster models and outputs
USR_CLUSTER_MOD = join(USR_CLUSTER_DIR, 'model-KMeans-{}-0.model'.format(NUM_USR_PROF_CLUSTER))
TWEET_TOPIC_DIR = join(DATA_ROOT, 'tweet-topic') #directory to contain topic models and outputs for tweets
TWEET_SENTI_TRAIN_TEXT=join(DATA_ROOT,'tweet-senti-text.txt') #file containing text for training sentiment model

if not os.path.exists(TWEET_TOPIC_DIR):
	os.makedirs(TWEET_TOPIC_DIR)
if not os.path.exists(USR_CLUSTER_DIR):
	os.makedirs(USR_CLUSTER_DIR)
if not os.path.exists(PRIV_TWEET_MODEL_PATH):
    os.makedirs(PRIV_TWEET_MODEL_PATH)


def get_profile_cluster(profile_desc):
    '''Assigns profile cluster based on the mode of
    predicted cluster across all words in profile_desc'''
    clusters = []
    for word in profile_desc.split():
        clusters.append(prof_cluster_model.predict(
            prof_ft_model.get_word_vector(word).reshape(1,-1)))
    return mode(clusters)[0][0]

def get_text_topic(text, topic_words):
    '''Assigns topic cluster to text based on the number of
        topic words are present in it'''
    idx= np.zeros(len(topic_words))
    for t in range(len(topic_words)):
        for word in topic_words[t]:
            if word in text:
                idx[t]+=1
    if np.max(idx) == 0: #if no topic words are found in text
        return -1
    return np.argmax(idx)

def get_text_topic_cos(text, topic_texts, ft_model):
    '''Assigns topic cluster to text based on its cosine 
        distance from all topics'''
    #
    text_vec = ft_model.get_sentence_vector(text)
    cos_sims = []
#
    for topic_text in topic_texts:
        topic_vec = ft_model.get_sentence_vector(topic_text)
        cos_sim = np.dot(text_vec, topic_vec)/(np.linalg.norm(text_vec)*np.linalg.norm(topic_vec))
        cos_sims.append(cos_sim)
#
    return (np.argmax(cos_sims), np.max(cos_sims))

def get_topic_words(file):
    '''returns list of list of words and list of sentences for topics'''
    topics = [line.split(':')[1] for line in open(file)]
    topic_words = []
    topic_texts = []
    for line in topics:
        words = [w.strip() for w in line.split(' ') if len(w.strip())>2]
        topic_words.append(words)
        topic_texts.append(' '.join(words))
    return topic_words, topic_texts

print('#################### Concatenating all data frames ####################\n')
os.system('python3 main.py --task clean-csv-dfs --in_dir {} --out_dir {} --kw_file {} --stat_dir {}'.format(
            join(DATA_ROOT, 'ed-tweets'), join(DATA_ROOT, 'ed-tweet-dfs'), 'donetags.csv', 'tweet-stat'))
os.system('python3 main.py --task merge-dfs --in_dir {} --out_dir {}'.format(
            join(DATA_ROOT, 'ed-tweet-dfs'), DATA_ROOT))

print('\n\n#################### Handling profile data ####################\n')
print('\tpreprocessing profile text...')
preprocess_tweet_data.preprocess_user_df(infile=TWEET_USER_DF, outfile=TWEET_USER_DF)

print('\tsaving profile dataframe containing en users...')
user_df = pd.read_pickle(TWEET_USER_DF)
user_df_en = user_df[user_df.profile_lang=='en']
user_df_en.to_pickle(TWEET_USER_DF_EN)
user_df_en = user_df_en[~user_df_en.index.duplicated(keep='first')] 
user_df_en.to_pickle(TWEET_USER_DF_EN_UNIQUE)

user_df_en['profile_desc_clean'] = user_df_en.progress_apply(lambda u: 
                preprocess_tweet_data.preprocess_profile_desc(
                    profile= u.profile_desc, lemmatize=True) \
                if isinstance(u.profile_desc, str) else '', axis=1)


print('\twriting english profile text in file...')
with open(TWEET_USER_PROFILE_TEXT, 'w') as file:
    user_df_en.progress_apply(lambda row: file.write(row.profile_desc_clean+'\n'), axis=1)
    file.close()


print('\ttraining fastText skipgram model with profile description...')
os.system('./helpers/fastText/fasttext skipgram -input {} -output {}'.format(TWEET_USER_PROFILE_TEXT, PROFILE_FT_MODEL))

print('\ttraining model to cluster profiles...')
os.system('python3 -Xfaulthandler helpers/cluster-analysis/main.py \
            --entities fasttext \
            --ftmodel {} \
            --clustering_algo {} \
            --vocab {} \
            --num_topics {}  \
            --rerank tf  \
            --stopwords_file helpers/stopwords-en.txt \
            --weight_file {} --weight_type WGT \
            --tfidf_file {} \
            --model_path {} > profile-cluster'.format(
                PROFILE_FT_MODEL+'.bin', ALGO_USR_PROF_CLUSTER, 
                TWEET_USER_PROFILE_TEXT, NUM_USR_PROF_CLUSTER, 
                PROFILE_WORD_WEIGHTS, 
                PROFILE_WORD_TFIDF, USR_CLUSTER_DIR))

print('\tpredicting profile clusters using trained model')

prof_ft_model = ft.load_model(PROFILE_FT_MODEL+'.bin')
prof_cluster_model = pickle.load(open(USR_CLUSTER_MOD, 'rb'))


# user_df_en['profile_cluster_w'] = user_df_en.progress_apply(lambda user: 
#                 get_profile_cluster(user.profile_desc_clean) \
#                 if len(user.profile_desc_clean.strip())>3 else -1, axis=1)

# user_df_en['profile_cluster_s'] = user_df_en.progress_apply(lambda user: 
# 				prof_cluster_model.predict(
#                 prof_ft_model.get_sentence_vector(
#                 user.profile_desc_clean).reshape(1,-1))[0] \
#                 if len(user.profile_desc_clean.strip())>3 else -1, axis=1)

profile_words, profile_sentences = get_topic_words('user-profile-words.txt')
# user_df_en['profile_cluster'] = user_df_en.progress_apply(lambda u: 
#                     get_text_topic(u.profile_desc_clean, profile_words) \
#                     if len(u.profile_desc_clean.strip())>=3 else -1, axis=1)

user_df_en['profile_cluster_cos'] = user_df_en.progress_apply(lambda u: 
                    get_text_topic_cos(u.profile_desc_clean, profile_sentences, prof_ft_model) \
                    if len(u.profile_desc_clean.strip())>=3 else (-1,0), axis=1)
user_df_en['profile_cluster'] = user_df_en.progress_apply(lambda row: 
                    row.profile_cluster_cos[0] if type(row.profile_cluster_cos)!=int else -1, axis=1)
user_df_en['profile_cluster_con'] = user_df_en.progress_apply(lambda row: 
                    row.profile_cluster_cos[1] if type(row.profile_cluster_cos)!=int else 0, axis=1) 

user_df_en.to_pickle(TWEET_USER_DF_EN_UNIQUE)

print('\n\n#################### Preprocessing tweets ####################\n')
tweet_df = pd.read_pickle(TWEET_DF_FILE)
tweet_df_en = tweet_df[tweet_df.lang=='en']
tweet_df_en['author_id'] = tweet_df_en.author_id.astype('str')

print('cleaning tweet text...')
# use the same preprocessing as for the sentiment 
tweet_df_en['text_clean_senti'] = tweet_df_en.progress_apply(lambda tweet:
                            preprocess_tweet_data.preprocess_tweet_senti(tweet.text, lemmatize=True) \
                            if isinstance(tweet.text, str) else ' ', axis=1)
print('writing clean tweet text in file...')
with open(CLEAN_TWEET_TEXT_EN, 'w') as file:
    tweet_df_en.progress_apply(lambda tweet:
        file.write(tweet.text_clean+'\n'), axis=1)
    file.close()

print('#################### Sentiment of Tweet Texts ####################\n')
# vader = SentimentIntensityAnalyzer()
# tweet_df_en['senti'] = tweet_df_en.progress_apply(lambda tweet: vader.polarity_scores(
#         ' '.join(process_tweet_data.senti_tokenizer.tokenize(tweet.text))), axis=1)
senti_model = ft.train_supervised(input=TWEET_SENTI_TRAIN_TEXT, dim=100, epoch=10, wordNgrams=2)
senti_model.save_model(DATA_ROOT+'senti_model.bin')
tweet_df_en['senti'] = tweet_df_en.progress_apply(lambda tweet: 
                    senti_model.predict(tweet.text_clean)[0][0].split('__')[-1], axis=1)
tweet_df_en.to_pickle(EN_TWEET_DF_FILE)

print('\n\n#################### Identifying security/privacy related tweets ####################\n')
sec_priv_kws = [line.strip() for line in  open('privacy_kws')]

def contains_any_keyword(text, kws):
    for kw in kws:
        if kw in text:
            return True
    return False

sec_priv_df = tweet_df_en[tweet_df_en.progress_apply(lambda tweet: 
                contains_any_keyword(tweet.text, sec_priv_kws), axis=1)]

sec_priv_df['text_clean_topic'] = sec_priv_df.progress_apply(lambda tweet:
                            preprocess_tweet_data.preprocess_tweet_topic(tweet.text, lemmatize=True) \
                            if isinstance(tweet.text, str) else ' ', axis=1)

sec_priv_df.to_pickle(PRIV_TWEET_DF)
print('\tclustering security-privacy related tweets')
with open(PRIV_TWEET_TEXT, 'w') as file:
    sec_priv_df.progress_apply(lambda tweet:
        file.write(tweet.text_clean_topic+'\n'), axis=1)
    file.close()


print('\ttraining fastText skipgram model with priv-sec tweets...')
os.system('./helpers/fastText/fasttext skipgram -input {} -output {} -dim 100'.format(PRIV_TWEET_TEXT, PRIV_FT_MODEL))
os.system('python3 -Xfaulthandler helpers/cluster-analysis/main.py \
            --entities fasttext \
            --ftmodel {}.bin\
            --clustering_algo {} \
            --vocab {}\
            --num_topics 5 8 10 15 20 25 30  \
            --rerank tf  \
            --stopwords_file helpers/stopwords-en.txt \
            --weight_file {}/priv-weights.wgt --weight_type WGT \
            --tfidf_file {}/priv-tfidf \
            --model_path {}'.format(
                PRIV_FT_MODEL, ALGO_USR_PROF_CLUSTER, PRIV_TWEET_TEXT,
                PRIV_TWEET_MODEL_PATH, PRIV_TWEET_MODEL_PATH, PRIV_TWEET_MODEL_PATH))

priv_ft_model = ft.load_model(PRIV_FT_MODEL+'.bin')
priv_cluster_model = pickle.load(open(PRIV_TWEET_MODEL_PATH+'model-KMeans-10-0.model', 'rb'))



priv_topic_words, priv_topic_sentences = get_topic_words('priv-tweet-topics.txt')
sec_priv_df['topic'] = sec_priv_df.progress_apply(lambda tweet: 
                get_text_topic(tweet.text_clean_topic, priv_topic_words) \
                if len(tweet.text_clean_topic.strip())>3 else -1, axis=1)

topic_cos = sec_priv_df.progress_apply(lambda tweet: 
                get_text_topic_cos(tweet.text_clean_topic, priv_topic_sentences, priv_ft_model) \
                if len(tweet.text_clean_topic.strip())>3 else (-1, 0), axis=1)
sec_priv_df['topic_cos'] = topic_cos.apply(lambda r: r[0])
sec_priv_df['topic_con'] = topic_cos.apply(lambda r: r[1])

sec_priv_df['senti'] = sec_priv_df.progress_apply(lambda tweet: 
                    senti_model.predict(tweet.text_clean_senti)[0][0].split('__')[-1], axis=1)

sec_priv_df.to_pickle(PRIV_TWEET_DF)

# print('#################### Preprocessing Tweet Texts ####################\n')
# text_processing.create_preprocessed_tweet_data(outfile=EN_TWEET_DF_FILE, data_frame_file=EN_TWEET_DF_FILE)



# print('creating trigrams...')
# topic_modeling.create_trigrams(textfile=preprocessed_tweet_file, outpath=topic_modeling_dir)

# print('creating dictionary and corpus..')
# _,_ = topic_modeling.create_dictionary_and_corpus(text_corpus_file=join(topic_modeling_dir,'trigrams.txt'), outpath=topic_modeling_dir)

# print('training and evaluating lda models..')
# topic_modeling.compute_coherence_values(
#                 preprocessed_text_file = join(topic_modeling_dir,'trigrams.txt'), 
#                 dictionary_file=join(topic_modeling_dir,'dictionary.pkl'),  
#                 corpus_file=join(topic_modeling_dir,'corpus.pkl'), 
#                 out_dir = join(topic_modeling_dir,'ldamodels'), 
#                 limit=500, start=50, step=50)

