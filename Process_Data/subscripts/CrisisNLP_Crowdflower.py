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

    directory = "../Data/CrisisNLP_labeled_data_crowdflower_v2/CrisisNLP_labeled_data_crowdflower"

    if not os.path.exists(data_dir):
        data = {}
    else:
        with open(data_dir) as file:
            data = json.load(file)

    label_map = {'missing_trapped_or_found_people': 'casaulties and damage',
                 'infrastructure_and_utilities_damage': 'casaulties and damage',
                 'donation_needs_or_offers_or_volunteering_services': 'donation',
                 'caution_and_advice': 'caution and advice',
                 'displaced_people_and_evacuations': 'casaulties and damage',
                 'treatment': 'unlabeled',
                 'disease_transmission': 'unlabeled',
                 'sympathy_and_emotional_support': 'not informative',
                 'other_useful_information': 'unlabeled',
                 'deaths_reports': 'casaulties and damage',
                 'prevention': 'caution and advice',
                 'not_related_or_irrelevant': 'not informative',
                 'affected_people': 'casaulties and damage',
                 'injured_or_dead_people': 'casaulties and damage',
                 'disease_signs_or_symptoms': 'casaulties and damage'}

    disaster_keys = ["Pakistan_Earthquake", "California_Earthquake", "Chile_Earthquake",
                     "Ebola", "Hurricane_Odile", "India_Floods",
                     "Middle_East_Respiratory_Syndrome", "Pakistan_Floods", "Typhoon_Hagupit",
                     "Cyclone_Pam", "Nepal_Earthquake"]

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

                    if "pakistan_eq" in filename.lower():
                        disaster_key = "Pakistan_Earthquake"
                    elif "mers" in filename.lower():
                        disaster_key = "Middle_East_Respiratory_Syndrome"

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
                            if i > 0 and len(row) >= 3:
                                data[disaster_key]["tweet_ids"].append(row[0])
                                tweet = str(row[1])
                                tweet = tweet.encode("ascii", errors="ignore").decode()
                                if len(tweet.split(" ")) > 300:
                                    print("Crowdflower1: "+" ".join(tweet))
                                data[disaster_key]["tweets"].append(tweet)
                                data[disaster_key]["labels"].append(row[2])
                                data[disaster_key]["simple_labels"].append(label_map[row[2]])
                                data[disaster_key]["source"].append("CrisisNLP_Crowdflower")

                                """
                                labels.append(row[2])


                                if row[2] == 'disease_signs_or_symptoms':
                                    print(tweet)
                                    print('\n\n')
                                """

    """

    display_step = 100

    for disaster_key in disaster_keys:

        tweet_ids = data[disaster_key]["tweet_ids"]

        tweets = data[disaster_key]["tweets"]

        labels = data[disaster_key]["labels"]

        simple_labels = data[disaster_key]["simple_labels"]

        print("\n\n{}\n\n".format(disaster_key))

        i = 0

        for tweet_id, tweet, label, simple_label in zip(tweet_ids, tweets, labels, simple_labels):
            if i % display_step == 0:
                print("tweet_id: ", tweet_id)
                print("tweet: ", tweet)
                print("label: ", label)
                print("simple label: ", simple_label)
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
