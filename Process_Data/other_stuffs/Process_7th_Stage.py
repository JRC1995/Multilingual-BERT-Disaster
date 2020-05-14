import numpy as np
import re
import random
import string
import math
import csv
import codecs
import os
import json

filename = "../Data/training.1600000.processed.noemoticon.csv"

data_dir = "../Processed_Data/Processed_Data_Intermediate_Sentiment.json"

if not os.path.exists(data_dir):
    data = {}
else:
    with open(data_dir) as file:
        data = json.load(file)

labels = []
tweets = []
tweet_ids = []

count_0 = 0
count_2 = 0
count_4 = 0

with codecs.open(filename, 'r', encoding='ISO 8859-1') as csvfile:

    csv_reader = csv.reader(csvfile)

    for i, row in enumerate(csv_reader):
        tweet_id = str(row[1])
        tweet = str(row[5])
        label = int(row[0])

        if label == 4:
            label = 1
        else:
            label = 0

        tweet_ids.append(tweet_id)
        tweets.append(tweet)
        labels.append(label)

"""

display_step = 100

for disaster_key in disaster_keys:

    tweet_ids = data[disaster_key]["tweet_ids"]

    tweets = data[disaster_key]["tweets"]

    labels = data[disaster_key]["labels"]

    print("\n\n{}\n\n".format(disaster_key))

    i = 0

    for tweet_id, tweet, label in zip(tweet_ids, tweets, labels):
        if i % display_step == 0:
            print("tweet_id: ", tweet_id)
            print("tweet: ", tweet)
            print("label: ", label)
            print("\n\n")
"""

# print(set(labels))

data = {}
data["tweet_ids"] = tweet_ids
data["tweets"] = tweets
data["labels"] = labels

with open(data_dir, 'w') as outfile:
    json.dump(data, outfile)


# process()
