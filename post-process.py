import pandas as pd
import glob, sys, csv

def combine_dfs(indir, tag, outdir):
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


indir = sys.argv[1]
tag_file = sys.argv[2]
outdir = sys.argv[3]

tags = set([])
with open(tag_file) as csv_file:
	reader = csv.reader(csv_file)
	tags = set(list(reader)[0])

for tag in tags:
	combine_dfs(indir, tag, outdir)