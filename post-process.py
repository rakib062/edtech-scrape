import pandas as pd
import glob, sys, csv, json

def combine_dfs(indir, tag, outdir, stat_dir):
	'''
	Combine all dataframes for a given hastag to one dataframe.
	'''

	print("Tag: ", tag)
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
	df.to_csv('{}/tweets-search-{}.csv'.format(outdir, tag))

	df['tweetid'] = df.tweetid.astype(int)
	latest_tweet = df[df.tweetid>=df.tweetid.max()].iloc[0]
	print(latest_tweet)
	print(latest_tweet.to_json())
	with open('{}/tweet-stat-{}.json'.format(stat_dir, tag), 'w') as fp:
		json.dump(latest_tweet.to_json(), fp)
	# with open('{}/tweet-stat-{}.csv'.format(stat_dir, tag), 'w') as csv_file:
	# 	writer = csv.writer(csv_file)
	# 	writer.writerow([latest_tweet.tweetid, latest_tweet.created_at])


	dfs = [pd.read_csv(file, lineterminator='\n') for file in \
			glob.glob('{}/users-search-{}*.csv'.format(indir, tag))]
	
	if len(dfs)>0:
		pd.concat(dfs).to_csv('{}/users-search-{}.csv'.format(outdir, tag))

	dfs = [pd.read_csv(file, lineterminator='\n') for file in \
			glob.glob('{}/inc-tweets-search-{}*.csv'.format(indir, tag))]
	if len(dfs)>0:
		df = pd.concat(dfs)
		df['tweetid'] = df.tweetid.astype(str)
		df = df[df.tweetid!='nan']
		df.to_csv('{}/inc-tweets-search-{}.csv'.format(outdir, tag))

	dfs = [pd.read_csv(file, lineterminator='\n') for file in \
			glob.glob('{}/media-search-{}*.csv'.format(indir, tag))]
	if len(dfs)>0:
		df = pd.concat(dfs)
		#df['tweetid'] = df.tweetid.astype(str)
		#df = df[df.tweetid!='nan']
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
