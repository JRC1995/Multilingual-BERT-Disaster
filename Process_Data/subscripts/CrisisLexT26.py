import numpy as np
import re
import random
import string
import math
import csv
from langdetect import detect
import codecs
import os
import json


def process(data_dir, data_dir_public):

    directory = "../Data/CrisisLexT26-v1.0/CrisisLexT26/"

    if not os.path.exists(data_dir):
        data = {}
    else:
        with open(data_dir) as file:
            data = json.load(file)

    disaster_keys = ["Colorado_Wildfires", "Costa_Rica_Earthquake", "Guatemala_Earthquake", "Italy_Earthquake",
                     "Philipinnes_Floods", "Typhoon_Pablo", "Venezuela_Refinery", "Alberta_Floods",
                     "Australia_Bushfire", "Bohol_Earthquake", "Boston_Bombings",
                     "Brazil_Nightclub_Fire", "Colorado_Floods", "Glasglow_Helicopter_Crash",
                     "LA_Airport_Shootings", "Lac_Megantic_Train_Crash", "Manila_Floods",
                     "NY_Train_Crash", "Queensland_Floods", "Russia_Meteor", "Sardinia_Floods",
                     "Savar_Building_Collapse", "Singapore_Haze", "Spain_Train_Crash",
                     "Typhoon_Yolanda", "West_Texas_Explosion"]

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

    label_map = {'Not labeled': 'ignore',
                 'Affected individuals': 'casaulties and damage',
                 'Not applicable': 'not informative',
                 'Donations and volunteering': 'donation',
                 'Sympathy and support': 'not informative',
                 'Caution and advice': 'caution and advice',
                 'Infrastructure and utilities': 'casaulties and damage',
                 'Other Useful Information': 'unlabeled'}

    labels = []

    for root, dirs, files in os.walk(directory, topdown=False):
        for file in files:
            filename = os.path.join(root, file)

            if len(filename) > 4:
                if filename[-4:] == '.tsv' or filename[-4:] == '.csv':

                    if "period" not in filename:

                        print(filename)

                        for key in disaster_keys:
                            if key.lower() in filename.lower():
                                disaster_key = key

                        if filename[-4:] == '.tsv':
                            encoding = 'ISO 8859-1'
                        else:
                            encoding = 'utf-8'

                        with codecs.open(filename, 'r', encoding=encoding) as csvfile:

                            if filename[-3:] == 'tsv':
                                csv_reader = csv.reader(csvfile, delimiter='\t')
                            else:
                                csv_reader = csv.reader(csvfile)

                            for i, row in enumerate(csv_reader):
                                if i > 0:
                                    label = row[3]
                                    if label_map[row[3]] != 'ignore':
                                        data[disaster_key]["tweet_ids"].append(row[0])
                                        tweet = str(row[1])
                                        tweet = tweet.encode("ascii", errors="ignore").decode()
                                        if len(tweet.split(" ")) > 300:
                                            print("CrisisLexT26: "+" ".join(tweet))
                                        data[disaster_key]["tweets"].append(tweet)
                                        data[disaster_key]["labels"].append(row[3])
                                        data[disaster_key]["simple_labels"].append(
                                            label_map[row[3]])
                                        data[disaster_key]["source"].append("CrisisLexT26")
                                        # labels.append(row[3])

    # print(set(labels))

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

    with open(data_dir, 'w') as outfile:
        json.dump(data, outfile)

    with open(data_dir_public, 'w') as outfile:
        for disaster_key in data:
            del data[disaster_key]["tweets"]
        json.dump(data, outfile)


# process()
