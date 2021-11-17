import pandas as pd
import glob, sys

def combine_dfs(indir, tag, outdir):
	'''
	Combine all dataframes for a given hastag to one dataframe.
	'''
	dfs = [pd.read_csv(file, lineterminator='\n') for file in \
			glob.glob('{}/tweets-search-{}*.csv'.format(indir, tag))]

	print("tweet dataframes:{}, tweets:{}".format(len(dfs), 
		sum([len(df) for df in dfs])))
	if len(dfs)>0:
		pd.concat(dfs).to_csv('{}/tweets-search-{}.csv'.format(outdir, tag))

	dfs = [pd.read_csv(file, lineterminator='\n') for file in \
			glob.glob('{}/users-search-{}*.csv'.format(indir, tag))]
	
	if len(dfs)>0:
		pd.concat(dfs).to_csv('{}/users-search-{}.csv'.format(outdir, tag))

	dfs = [pd.read_csv(file, lineterminator='\n') for file in \
			glob.glob('{}/inc-tweets-search-{}*.csv'.format(indir, tag))]
	if len(dfs)>0:
		pd.concat(dfs).to_csv('{}/inc-tweets-search-{}.csv'.format(outdir, tag))

	dfs = [pd.read_csv(file, lineterminator='\n') for file in \
			glob.glob('{}/media-search-{}*.csv'.format(indir, tag))]
	if len(dfs)>0:
		pd.concat(dfs).to_csv('{}/media-search-{}.csv'.format(outdir, tag))

combine_dfs(sys.argv[1], "#{}".format(sys.argv[2]), sys.argv[3])