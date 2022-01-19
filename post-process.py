import pandas as pd
import glob, sys, csv, json

def combine_dfs(indir, tag, outdir, stat_dir):
	'''
	Combine all dataframes for a given hastag to one dataframe.
	'''

	print("********************Tag: {}***********************".format(tag))
	dfs = [pd.read_csv(file, lineterminator='\n') for file in \
			glob.glob('{}/tweets-search-{}*.csv'.format(indir, tag))]
	if len(dfs)==0:    
		print("No data found.")
		return
	print("tweet dataframes:{}, tweets:{}, combined shape:{}".format(len(dfs), 
		sum([len(df) for df in dfs]), pd.concat(dfs).shape))

	df = pd.concat(dfs)
	df['tweetid'] = df.tweetid.astype(str)
	df = df[df.tweetid!='nan']
	df.set_index("tweetid", inplace=True)
	df[~df.index.duplicated(keep='first')]
	print("number of unique tweets: {}".format(len(df)))
	df.to_csv('{}/tweets-search-{}.csv'.format(outdir, tag))

	df.reset_index(inplace=True, drop=False)
	df['tweetid'] = df.tweetid.astype(int)
	latest_tweet = df[df.tweetid>=df.tweetid.max()].iloc[0]

	with open('tweet_count.txt', 'a') as f:
		f.write('tag: {}, count: {}\n'.format(tag, len(df)))
	
	with open('{}/tweet-stat-{}.json'.format(stat_dir, tag), 'w') as fp:
		json.dump(latest_tweet.to_json(), fp)

	dfs = [pd.read_csv(file, lineterminator='\n') for file in \
			glob.glob('{}/users-search-{}*.csv'.format(indir, tag))]
	
	if len(dfs)>0:
		df = pd.concat(dfs)
		df['userid'] = df.userid.astype(str)
		df = df[df.userid!='nan']
		df.set_index(userid, inplace=True)
		df[~df.index.duplicated(keep='first')]
		print("number of unique users: {}".format(len(df)))
		df.to_csv('{}/users-search-{}.csv'.format(outdir, tag))

	dfs = [pd.read_csv(file, lineterminator='\n') for file in \
			glob.glob('{}/inc-tweets-search-{}*.csv'.format(indir, tag))]
	if len(dfs)>0:
		df = pd.concat(dfs)
		df['tweetid'] = df.tweetid.astype(str)
		df = df[df.tweetid!='nan']
		df.set_index(tweetid, inplace=True)
		df[~df.index.duplicated(keep='first')]
		print("number of inc. unique tweets: {}".format(len(df)))
		df.to_csv('{}/inc-tweets-search-{}.csv'.format(outdir, tag))

	dfs = [pd.read_csv(file, lineterminator='\n') for file in \
			glob.glob('{}/media-search-{}*.csv'.format(indir, tag))]
	if len(dfs)>0:
		df = pd.concat(dfs)
		df['media_key'] = df.media_key.astype(str)
		df = df[df.media_key!='nan']
		df.set_index(media_key, inplace=True)
		df[~df.index.duplicated(keep='first')]
		df.to_csv('{}/media-search-{}.csv'.format(outdir, tag))


indir = sys.argv[1]
tag_file = sys.argv[2]
outdir = sys.argv[3]
stat_dir=sys.argv[4]

tags = set([])
with open(tag_file) as csv_file:
	reader = csv.reader(csv_file)
	tags = set(list(reader)[0])

for tag in tags:
	combine_dfs(indir, tag, outdir, stat_dir)
