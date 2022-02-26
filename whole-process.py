import os, sys, pickle, re
from os.path import join
import pandas as pd
import fasttext as ft
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
tqdm.pandas()
from scipy.stats import mode

sys.path.append('helpers/')
from tweetokenize import *

import text_processing
import preprocess_tweet_data

import config

DATA_ROOT = config.DATA_ROOT

NUM_USR_PROF_CLUSTER=100  
ALGO_USR_PROF_CLUSTER='KMeans'


TWEET_USER_DF = join(DATA_ROOT, 'users-all.pkl') #containing profile info for all users
TWEET_USER_DF_EN = join(DATA_ROOT, 'users-en.pkl') #containing profile info for users with (detected) 'en' description
TWEET_USER_PROFILE_TEXT = join(DATA_ROOT, 'tweet-profile-text-en.txt') #file containing cleaned profile descriptions
CLEAN_TWEET_TEXT_EN = join(DATA_ROOT,'tweet-text-en.txt') #file containing cleaned tweet text
PROFILE_FT_MODEL = join(DATA_ROOT, 'ft-profile-sg') #fasttext model for profle desc, using skipgram to catch semantic relationships
TWEET_DF_FILE = join(DATA_ROOT, 'tweets-all.pkl')
EN_TWEET_DF_FILE = join(DATA_ROOT,'tweets-en.pkl') #dataframe with english tweets 
PRIV_TWEET_DF = join(DATA_ROOT,'tweets-priv.pkl') #dataframe containing sec/priv related tweets 
PRIV_TWEET_TEXT = join(DATA_ROOT,'tweet-text-priv.txt') #file containing sec/priv tweet texts 
PRIV_TWEET_MODEL_PATH=join(DATA_ROOT, 'priv-tweet-models/') #directory for ft and cluster models for sec/priv related tweets
PRIV_FT_MODEL  = join(PRIV_TWEET_MODEL_PATH, 'ft-priv-tweet-sg/') #ft model name for sec/priv related tweets
USR_CLUSTER_DIR = join(DATA_ROOT, 'profile-cluster-models') #directory to save profile cluster models and outputs
USR_CLUSTER_MOD = join(USR_CLUSTER_DIR, 'model-KMeans-{}-0.model'.format(NUM_USR_PROF_CLUSTER))
TWEET_TOPIC_DIR = join(DATA_ROOT, 'tweet-topic') #directory to contain topic models and outputs for tweets

if not os.path.exists(TWEET_TOPIC_DIR):
	os.makedirs(TWEET_TOPIC_DIR)
if not os.path.exists(USR_CLUSTER_DIR):
	os.makedirs(USR_CLUSTER_DIR)
if not os.path.exists(PRIV_TWEET_MODEL_PATH):
    os.makedirs(PRIV_TWEET_MODEL_PATH)


def get_profile_cluster(profile_desc):
    clusters = []
    for word in profile_desc.split():
        clusters.append(prof_cluster_model.predict(
            prof_ft_model.get_word_vector(word).reshape(1,-1)))
    return mode(clusters)[0][0]

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

print('\twriting english profile text in file...')
preprocess_tweet_data.write_user_profile_des(infile=TWEET_USER_DF_EN, outfile=TWEET_USER_PROFILE_TEXT)


print('\ttraining fastText skipgram model with profile description...')
os.system('./helpers/fastText/fasttext skipgram -input {} -output {}'.format(TWEET_USER_PROFILE_TEXT, PROFILE_FT_MODEL))

print('\ttraining model to cluster profiles...')
os.system('python3 -Xfaulthandler helpers/cluster-analysis/main.py \
            --entities fasttext \
            --ftmodel data/ft-profile-sg.bin \
            --clustering_algo {} \
            --vocab data/tweet-profile-text.txt \
            --num_topics {}  \
            --rerank tf  \
            --stopwords_file helpers/stopwords-en.txt \
            --weight_file data/weights.wgt --weight_type WGT \
            --tfidf_file data/tfidf \
            --model_path data/profile-cluster-models > clustering.out'.format(ALGO_USR_PROF_CLUSTER, NUM_USR_PROF_CLUSTER))

print('\tpredicting profile clusters using trained model')

prof_ft_model = ft.load_model(PROFILE_FT_MODEL+'.bin')
prof_cluster_model = pickle.load(open(USR_CLUSTER_MOD, 'rb'))

user_df_en['profile_desc_clean'] = user_df.progress_apply(lambda user: 
                re.sub(r'\W+', ' ', str(user.profile_desc)) if type(user.profile_desc)==str else '', axis=1)

user_df_en['profile_cluster_w'] = user_df_en.progress_apply(lambda user: 
                get_profile_cluster(user.profile_desc_clean) \
                if len(user.profile_desc_clean.strip())>3 else -1, axis=1)

user_df_en['profile_cluster_s'] = user_df_en.progress_apply(lambda user: 
				prof_cluster_model.predict(
                prof_ft_model.get_sentence_vector(
                user.profile_desc_clean).reshape(1,-1))[0] \
                if len(user.profile_desc_clean.strip())>3 else -1, axis=1)


print('\n\n#################### Preprocessing tweets ####################\n')
tweet_df = pd.read_pickle(TWEET_DF_FILE)
tweet_df_en = tweet_df[tweet_df.lang=='en']

print('cleaning tweet text...')
tweet_df_en['clean_text'] = tweet_df_en.progress_apply(lambda tweet:
                            ' '.join(text_processing.preprocess_tweet(tweet)), axis=1)
print('writing clean tweet text in file...')
with open(CLEAN_TWEET_TEXT_EN, 'w') as file:
    tweet_df_en.progress_apply(lambda tweet:
        file.write(tweet.clean_text+'\n'), axis=1)
    file.close()

print('#################### Sentiment of Tweet Texts ####################\n')
senti_tokenizer = Tokenizer(usernames="", urls='', hashtags=False, phonenumbers='', 
                times='', numbers='', ignorequotes=True, ignorestopwords=True) 
vader = SentimentIntensityAnalyzer()

tweet_df_en['senti'] = tweet_df_en.progress_apply(lambda tweet: vader.polarity_scores(
        ' '.join(senti_tokenizer.tokenize(tweet.text))), axis=1)
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
sec_priv_df.to_pickle(PRIV_TWEET_DF)
print('\tclustering security-privacy related tweets')
with open(PRIV_TWEET_TEXT, 'w') as file:
    sec_priv_df.progress_apply(lambda tweet:
        file.write(tweet.clean_text+'\n'), axis=1)
    file.close()


print('\ttraining fastText skipgram model with priv-sec tweets...')
os.system('./helpers/fastText/fasttext skipgram -input {} -output {}'.format(PRIV_TWEET_TEXT, PRIV_FT_MODEL))
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
priv_cluster_model = pickle.load(open(PRIV_TWEET_MODEL_PATH+'model-KMeans-20-0.model', 'rb'))

def get_text_topic(text, topic_words):
    idx= np.zeros(len(topic_words))
    for word in text.split():
        for t in range(len(topic_words)):
            if word in topic_words[t]:
                idx[t]+=1
    return np.argmax(idx)


priv_topics = [line.split(':')[1] for line in open('priv-tweet-topics.txt')]
priv_topic_words = []
for line in priv_topics:
    words = [w.strip() for w in line.split(' ') if len(w.strip())>2]
    priv_topic_words.append(words)



sec_priv_df['topic'] = sec_priv_df.progress_apply(lambda tweet: 
                get_text_topic(tweet.clean_text, priv_topic_words) \
                if len(tweet.clean_text.strip())>3 else -1, axis=1)

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

