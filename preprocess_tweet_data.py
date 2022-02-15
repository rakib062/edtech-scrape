'''
code to prepare datasets
'''

import pandas as pd
import sys
sys.path.append('helpers/')
import text_processing
import re
import glob, sys,os
from tqdm import tqdm

data_path = './data/'
if not os.path.exists(data_path):
    os.makedirs(data_path)

def detect_profile_lang(lang1, lang2, lang3):
	if lang1==lang2:
		return lang1
	if lang1==lang3:
		return lang1
	if lang2==lang3:
		return lang2
	return 'NA'


def combine_dfs(src_path, dest_path):
	'''
	Combine tweet, user, etc. dataframes
	'''

	print('combining user dataframes...')	
	files = glob.glob(src_path+'/users-search-*.pkl')

	all_df=[]
	for i in tqdm(range(len(files))):
		df = pd.read_pickle(files[i])
		df.reset_index(inplace=True)
		df.set_index("userid", inplace=True)
		df[~df.index.duplicated(keep='first')]
		df['search_term'] =  os.path.basename(file[i]).split('.')[0].split('-')[-1]
		all_df.append(df)

	all_df = pd.concat(all_df)
	all_df.to_pickle(dest_path+'/ed-users-all.pkl')

	print('combining tweet dataframes...')	
	files = glob.glob(src_path+'/tweets-search-*.pkl')

	all_df=[]
	for i in tqdm(range(len(files))):
		df = pd.read_pickle(files[i])
		df.reset_index(inplace=True)
		df.set_index("tweetid", inplace=True)
		df[~df.index.duplicated(keep='first')]
		df = df[df.apply(lambda tweet: tweet.text.startswith("RT")==False, axis=1 )]
		df['search_term'] =  os.path.basename(file[i]).split('.')[0].split('-')[-1]
		all_df.append(df)

	all_df = pd.concat(all_df)
	all_df.to_pickle(dest_path+'/ed-tweets-all.pkl')




def preprocess_user_df():
	user_df = pd.read_pickle(data_path+'/ed-users-all.pkl')

	user_df['profile_desc_clean']= user_df.apply(lambda row: re.sub(r'\W+', ' ', str(row.profile_desc)).lower().strip(), axis=1)

	user_df['lang1'] = user_df.apply(lambda row: text_processing.detect_lang(row.profile_desc_clean, detector='langid') \
									if len(row.profile_desc_clean)>3 else 'NA', axis=1)
	user_df['lang2'] = user_df.apply(lambda row: text_processing.detect_lang(row.profile_desc_clean, detector='fasttext') \
									if len(row.profile_desc_clean)>3 else 'NA', axis=1)
	user_df['lang3'] = user_df.apply(lambda row: text_processing.detect_lang(row.profile_desc_clean, detector='cld3') \
									if len(row.profile_desc_clean)>3 else 'NA', axis=1)
	
	user_df['profile_lang'] = user_df.apply(lambda row: detect_profile_lang(row.lang1, row.lang2, row.lang3), axis=1)
	user_df.to_pickle(data_path+'/ed-users-all-preprocessed.pkl')