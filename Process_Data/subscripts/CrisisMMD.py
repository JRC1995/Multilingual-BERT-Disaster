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

    directory = "../Data/CrisisMMD_v1.0/annotations/"

    if not os.path.exists(data_dir):
        data = {}
    else:
        with open(data_dir) as file:
            data = json.load(file)

    disaster_keys = ["California_Wildfires", "Hurricane_Harvey", "Hurricane_Irma",
                     "Hurricane_Maria", "Iraq_Iran_Earthquake", "Mexico_Earthquake",
                     "Srilanka_Floods"]

    label_map = {'other_relevant_information': 'unlabeled',
                 'dont_know_or_cant_judge': 'ignore',
                 'affected_individuals': 'casaulties and damage',
                 'injured_or_dead_people': 'casaulties and damage',
                 'vehicle_damage': 'casaulties and damage',
                 'infrastructure_and_utility_damage': 'casaulties and damage',
                 'not_relevant_or_cant_judge': 'ignore',
                 'not_informative': 'not informative',
                 'rescue_volunteering_or_donation_effort': 'donation',
                 'missing_or_found_people': 'casaulties and damage'}

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

    for root, dirs, files in os.walk(directory, topdown=False):
        for file in files:
            filename = os.path.join(root, file)

            if len(filename) > 4:
                if filename[-4:] == '.tsv' or filename[-4:] == '.csv':

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
                                if len(row) > 12:
                                    if row[6] == '':
                                        label = row[2]
                                        if row[2] == '':
                                            label = 'unlabeled'
                                    else:
                                        label = row[6]

                                    if label_map[label] != 'ignore':
                                        data[disaster_key]["tweet_ids"].append(row[0])
                                        tweet = str(row[12])
                                        tweet = tweet.encode("ascii", errors="ignore").decode()
                                        if len(tweet.split(" ")) > 300:
                                            print("CrisisMMD: "+" ".join(tweet))
                                        data[disaster_key]["tweets"].append(tweet)

                                        data[disaster_key]["labels"].append(label)
                                        data[disaster_key]["simple_labels"].append(label_map[label])
                                        data[disaster_key]["source"].append("CrisisMMD")

                                        # labels.append(label)
    # print(set(labels))

    with open(data_dir, 'w') as outfile:
        json.dump(data, outfile)

    with open(data_dir_public, 'w') as outfile:
        for disaster_key in data:
            del data[disaster_key]["tweets"]
        json.dump(data, outfile)


# process()
