import numpy as np
import re
import random
import string
import math
import csv
import codecs
import os
import json


def process(data_dir, data_dir_public):

    filename = "../Data/deep-learning-for-big-crisis-data-master/data/sample.csv"

    if not os.path.exists(data_dir):
        data = {}
    else:
        with open(data_dir) as file:
            data = json.load(file)

    disaster_keys = ["Mixed"]

    for disaster_key in disaster_keys:

        if disaster_key not in data:
            data[disaster_key] = {}

        if "tweet_ids" not in data[disaster_key]:
            data[disaster_key]["tweet_ids"] = []

        if "tweets" not in data[disaster_key]:
            data[disaster_key]["tweets"] = []

        if "labels" not in data[disaster_key]:
            data[disaster_key]["labels"] = []

        if "simple_labels" not in data[disaster_key]:
            data[disaster_key]["simple_labels"] = []

        if "source" not in data[disaster_key]:
            data[disaster_key]["source"] = []

    labels = []

    label_map = {'Other Useful Information': 'unlabeled',
                 'Not related or irrelevant': 'not informative',
                 'Affected individuals': 'casaulties and damage',
                 'Sympathy and support': 'not informative',
                 'Donations and volunteering': 'donation',
                 'Infrastructure and utilities': 'casaulties and damage'}

    with codecs.open(filename, 'r') as csvfile:

        disaster_key = "Mixed"

        csv_reader = csv.reader(csvfile)

        for i, row in enumerate(csv_reader):
            if i > 0:
                data[disaster_key]["tweet_ids"].append(row[0])
                tweet = str(row[1])

                tweet = tweet.encode("ascii", errors="ignore").decode()
                if len(tweet.split(" ")) > 300:
                    print("BigCrisisData: "+" ".join(tweet))
                data[disaster_key]["tweets"].append(tweet)
                data[disaster_key]["labels"].append(row[2])
                data[disaster_key]["simple_labels"].append(label_map[row[2]])
                data[disaster_key]["source"].append("BigCrisisData")

                # labels.append(row[2])

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

    with open(data_dir, 'w') as outfile:
        json.dump(data, outfile)

    with open(data_dir_public, 'w') as outfile:
        for disaster_key in data:
            del data[disaster_key]["tweets"]
        json.dump(data, outfile)


# process()
