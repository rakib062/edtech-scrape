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
import senti_processing

import text_processing, topic_modeling
import preprocess_tweet_data

data_root ='/Users/admin/onedrive/work/edtech-scrape/data/'

NUM_USR_PROF_CLUSTER=50  
ALGO_USR_PROF_CLUSTER='KMeans'

TWEET_USER_DF = join(data_root, 'users-all.pkl') #containing profile info for all users
TWEET_USER_DF_EN = join(data_root, 'users-en.pkl') #containing profile info for users with (detected) 'en' description
TWEET_USER_PROFILE_TEXT = join(data_root, 'tweet-profile-text-en.txt') #file containing cleaned profile descriptions
PREPROCESSED_TWEET_TEXT = join(data_root,'tweet-text-en.txt') #file containing cleaned tweet text
PROFILE_FT_MODEL = join(data_root, 'ft-profile-sg') #fasttext model for profle desc, using skipgram to catch semantic relationships
TWEET_DF_FILE = join(data_root, 'tweets-all.pkl')
EN_TWEET_DF_FILE = join(data_root,'tweets-en.pkl') #dataframe with english tweets 
USR_CLUSTER_DIR = join(data_root, 'user-cluster') #directory to save profile cluster models and outputs
USR_CLUSTER_MOD = join(USR_CLUSTER_DIR, 'model-KMeans-10-0.model')
TWEET_TOPIC_DIR = join(data_root, 'tweet-topic') #directory to contain topic models and outputs for tweets

if not os.path.exists(TWEET_TOPIC_DIR):
	os.makedirs(TWEET_TOPIC_DIR)
if not os.path.exists(USR_CLUSTER_DIR):
	os.makedirs(USR_CLUSTER_DIR)

def get_profile_cluster(profile_desc):
    clusters = []
    for word in profile_desc.split():
        clusters.append(prof_cluster_model.predict(
            prof_ft_model.get_word_vector(word).reshape(1,-1)))
    return mode(clusters)[0][0]

print('#################### Concatenating all data frames ####################\n')
os.system('python3 main.py --task clean-csv-dfs --in_dir {} --out_dir {} --kw_file {} --stat_dir {}'.format(
            join(data_root, 'ed-tweets'), join(data_root, 'ed-tweet-dfs'), 'donetags.csv', join(data_root, 'tweet-stat')))
os.system('python3 main.py --task merge-dfs --in_dir {} --out_dir {}'.format(
            join(data_root, 'ed-tweet-dfs'), data_root))

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
            --model_outdir data/profile-cluster-models/ \
            --stopwords_file helpers/stopwords-en.txt \
            --weight_file data/weights.wgt --weight_type WGT \
            --tfidf_file data/tfidf \
            --model_path data/profile-cluster-models > clustering.out'.format(ALGO_USR_PROF_CLUSTER, NUM_USR_PROF_CLUSTER))

print('\tpredicting profile clusters using trained model')

prof_ft_model = ft.load_model(PROFILE_FT_MODEL+'.bin')
prof_cluster_model = pickle.load(open(USR_CLUSTER_MOD, 'rb'))

# user_df_en['profile_desc_clean'] = user_df.progress_apply(lambda user: 
#                 re.sub(r'\W+', ' ', str(user.profile_desc)) if type(user.profile_desc)==str else '', axis=1)

user_df_en['profile_cluster_w'] = user_df_en.progress_apply(lambda user: 
                get_profile_cluster(user.profile_desc_clean) \
                if len(user.profile_desc_clean.strip())>3 else -1, axis=1)

user_df_en['profile_cluster_s'] = user_df_en.progress_apply(lambda user: 
				prof_cluster_model.predict(
                prof_ft_model.get_sentence_vector(
                user.profile_desc_clean).reshape(1,-1))[0] \
                if len(user.profile_desc_clean.strip())>3 else -1, axis=1)


print('\n\n#################### Preprocessing tweets ####################\n')
tweet_df = pd.read_pickle(EN_TWEET_DF_FILE)
tweet_df_en = tweet_df[tweet_df.lang=='en']
tweet_df_en.to_pickle(EN_TWEET_DF_FILE)

print('\n\n#################### Identifying security/privacy related tweets ####################\n')
sec_priv_kws = [line.strip() for line in  open('privacy_kws')]

def contains_any_keyword(text, kws):
    for kw in kws:
        if kw in text:
            return True
    return False

sec_priv_df = tweet_df_en.progress_apply(lambda tweet: contains_any_keyword(tweet.text, sec_priv_kws), axis=1)
sec_priv_df = sec_priv_df[sec_priv_df==True]
sec_priv_df = tweet_df_en.loc[sec_priv_df.index]
sec_priv_df.to_pickle('data/sec_priv_df.pkl')

print('#################### Preprocessing Tweet Texts ####################\n')
text_processing.create_preprocessed_tweet_data(outfile=EN_TWEET_DF_FILE, data_frame_file=EN_TWEET_DF_FILE)

print('#################### Sentiment of Tweet Texts ####################\n')
vader = SentimentIntensityAnalyzer()
tweet_df['senti'] = tweet_df_en.progress_apply(lambda tweet: vader.polarity_scores(
        ' '.join(senti_tokenizer.tokenize(tweet.text))), axis=1)

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

