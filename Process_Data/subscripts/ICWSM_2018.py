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

    directory = "../Data/ACL_ICWSM_2018_datasets/ACL_ICWSM_2018_datasets/"

    if not os.path.exists(data_dir):
        data = {}
    else:
        with open(data_dir) as file:
            data = json.load(file)

    disaster_keys = ["Nepal_Earthquake", "Queensland_Floods"]

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

    label_map = {'relevant': 'unlabeled',
                 'not_relevant': 'not informative'}

    for root, dirs, files in os.walk(directory, topdown=False):
        for file in files:
            filename = os.path.join(root, file)

            if len(filename) > 4:
                if filename[-4:] == '.tsv' or filename[-4:] == '.csv':

                    print(filename)

                    if "nepal_earthquake" in filename.lower():
                        disaster_key = "Nepal_Earthquake"
                    else:
                        disaster_key = "Queensland_Floods"

                    encoding = 'ISO 8859-1'

                    with codecs.open(filename, 'r', encoding=encoding) as csvfile:

                        csv_reader = csv.reader(csvfile, delimiter='\t', quotechar='"')

                        for i, row in enumerate(csv_reader):
                            if i > 0:

                                tweet = str(row[1])
                                tweet = tweet.encode("ascii", errors="ignore").decode()

                                if len(tweet.split(" ")) > 40:
                                    row = "\t".join(row)
                                    subrows = row.split("\n")
                                    for subrow in subrows:
                                        subrow = subrow.split("\t")
                                        subtweet = str(subrow[1])
                                        subtweet = subtweet.encode(
                                            "ascii", errors="ignore").decode()
                                        #print("0", subtweet)
                                        #print("1", subrow[0])
                                        #print("2", subrow[2])
                                        if len(subtweet.split(" ")) < 40:
                                            data[disaster_key]["tweet_ids"].append(subrow[0])
                                            data[disaster_key]["tweets"].append(subtweet)
                                            data[disaster_key]["labels"].append(subrow[2])
                                            data[disaster_key]["simple_labels"].append(
                                                label_map[subrow[2]])
                                            data[disaster_key]["source"].append("ICWSM_2018")

                                else:

                                    data[disaster_key]["tweet_ids"].append(row[0])
                                    data[disaster_key]["tweets"].append(tweet)
                                    data[disaster_key]["labels"].append(row[2])
                                    data[disaster_key]["simple_labels"].append(label_map[row[2]])
                                    data[disaster_key]["source"].append("ICWSM_2018")
                                    # labels.append(row[2])

    # print(set(labels))

    with open(data_dir, 'w') as outfile:
        json.dump(data, outfile)

    with open(data_dir_public, 'w') as outfile:
        for disaster_key in data:
            del data[disaster_key]["tweets"]
        json.dump(data, outfile)


# process()
