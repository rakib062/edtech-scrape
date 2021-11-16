import sys, csv,os

outdir = sys.argv[1]
if not os.path.exists(outdir):
        os.makedirs(outdir)
        
with open('tags.csv') as csv_file:
    reader = csv.reader(csv_file)
    tags = set(list(reader)[0])
    
donetags = set([])
if os.path.isfile('donetags.csv'): 
    with open('donetags.csv') as csv_file:
        reader = csv.reader(csv_file)
        donetags = set(list(reader)[0])

tags = tags.difference(donetags)
i=1
for tag in tags:
    print("Starting search for tag no:{} of {}, tag:{}".format(i, len(tags),tag))
    search_tweets(tag, outdir)
    donetags.add(tag)
    with open('donetags.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        writer.writerow(donetags)
