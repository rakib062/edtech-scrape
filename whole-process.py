import os
import sys
from os.path import join
import pandas as pd
import fasttext as ft
from tqdm import tqdm
tqdm.pandas()


sys.path.append('helpers/')

import text_processing, topic_modeling
import preprocess_tweet_data

data_root ='./data/'

NUM_USR_PROF_CLUSTER=50  
ALGO_USR_PROF_CLUSTER='KMeans'

TWEET_USER_DF = join(data_root, 'ed-users-all.pkl') #containing profile info for all users
TWEET_USER_PROFILE_TEXT = join(data_root, 'tweet-profile-text.txt') #file containing cleaned profile descriptions
PREPROCESSED_TWEET_TEXT = join(data_root,'tweet-text-en.txt') #file containing cleaned tweet text
PROFILE_FT_MODEL = join(data_root, 'ft-profile-sg') #fasttext model for profle desc, using skipgram to catch semantic relationships
#PROFILE_VEC_DF = join(data_root,'profile-vec-df.pkl')

EN_TWEET_DF_FILE = join(data_root,'ed-tweets-all-en.pkl') #dataframe with english tweets 
USR_CLUSTER_DIR = join(data_root, 'user-cluster') #directory to save profile cluster models and outputs
TWEET_TOPIC_DIR = join(data_root, 'tweet-topic') #directory to contain topic models and outputs for tweets

if not os.path.exists(TWEET_TOPIC_DIR):
	os.makedirs(TWEET_TOPIC_DIR)
if not os.path.exists(USR_CLUSTER_DIR):
	os.makedirs(USR_CLUSTER_DIR)


# print('#################### Concatenating all data frames ####################\n')
# os.system('python combine-dataframes.py {} {} {} {}'.format(join(data_root, 'ed-tweet-data'), 'donetags.csv', ''))

print('\n\n#################### Handling profile data ####################\n')
print('\tpreprocessing profile text...')
preprocess_tweet_data.preprocess_user_df(infile=TWEET_USER_DF, outfile=TWEET_USER_DF)

print('\twriting profile text in file...')
preprocess_tweet_data.write_user_profile_des(infile=TWEET_USER_DF, outfile=TWEET_USER_PROFILE_TEXT)

print('\ttraining fastText skipgram model with profile description...')
os.system('./helpers/fastText/fasttext skipgram -input {} -output {}'.format(TWEET_USER_PROFILE_TEXT, PROFILE_FT_MODEL))

os.system('python3 helpers/cluster-analysis/main.py --entities fasttext --clustering_algo KMeans --vocab {} --num_topics 20 30 --doc_info WGT --rerank tf'.format(TWEET_USER_PROFILE_TEXT))

print('\ttraining model to cluster profiles...')
os.system('python -Xfaulthandler helpers/cluster-analysis/main.py \
            --entities fasttext --ftmodel data/ft-profile-sg.bin \
            --clustering_algo {} --vocab data/tweet-profile-text.txt \
            --num_topics {}  --rerank tf  --model_outdir data/profile-cluster-models/\
            --stopwords_file helpers/stopwords-en.txt \
            --weight_file data/weights.wgt --weight_type WGT \
            --tfidf_file data/tfidf'.format(ALGO_USR_PROF_CLUSTER, NUM_USR_PROF_CLUSTER))

print('\tpredicitn profile clusters using trained model')
user_df = pd.read_pickle(TWEET_USER_DF)
# prof_model = 





print('\n\n#################### Identifying security/privacy related tweets ####################\n')
sec_priv_kws = [line.strip() for line in  open('privacy_kws')]

def contains_any_keyword(text, kws):
    for kw in kws:
        if kw in text:
            return True
    return False

tweet_df = pd.read_pickle(EN_TWEET_DF_FILE)
sec_priv_df = tweet_df.progress_apply(lambda tweet: contains_any_keyword(tweet.text, sec_priv_kws), axis=1)
sec_priv_df = sec_priv_df[sec_priv_df==True]
sec_priv_df = tweet_df.loc[sec_priv_df.index]
sec_priv_df.to_pickle('data/sec_priv_df.pkl')

#################### Preprocessing Tweet Texts ####################
# print('#################### Preprocessing Tweet Texts ####################\n')
# text_processing.create_preprocessed_tweet_data(outfile=preprocessed_tweet_file, data_frame_file=tweet_df_file)

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

