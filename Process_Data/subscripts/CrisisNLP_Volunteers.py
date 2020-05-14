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

    directory = "../Data/CrisisNLP_volunteers_labeled_data/CrisisNLP_volunteers_labeled_data"

    if not os.path.exists(data_dir):
        data = {}
    else:
        with open(data_dir) as file:
            data = json.load(file)

    disaster_keys = ["California_Earthquake", "Chile_Earthquake",
                     "Ebola", "Hurricane_Odile", "Iceland_volcano",
                     "Malaysia_Airline_MH370", "Middle_East_Respiratory_Syndrome",
                     "Typhoon_Hagupit", "Cyclone_Pam", "Nepal_Earthquake",
                     "Landslides_Worldwide"]

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

    """

    : 'casaulties and damage'
    : 'caution and advice'
    : 'not informative'
    : 'donation'
    : 'unlabeled'
    : 'ignore'

    """

    label_map = {'Other relevant information': 'unlabeled',
                 'Displaced people': 'casaulties and damage',
                 'Yes': 'ignore',
                 'Needs of those affected': 'casaulties and damage',
                 'Donations of money': 'donation',
                 'Not related to crisis': 'not informative',
                 'Personal updates': 'ignore',
                 'Infrastructure': 'casaulties and damage',
                 'Shelter and supplies': 'donation',
                 'Personal only': 'ignore',
                 'Other relevant': 'unlabeled',
                 'Injured and dead': 'casaulties and damage',
                 'Animal management': 'ignore',
                 'Personal': 'ignore',
                 'Volunteer or professional services': 'donation',
                 'Physical landslide': 'ignore',
                 'Sympathy and emotional support': 'not informative',
                 'Infrastructure and utilities': 'casaulties and damage',
                 'Donations of supplies and/or volunteer work': 'donation',
                 'Not physical landslide': 'ignore',
                 'Not related or irrelevant': 'not informative',
                 'Non-government': 'ignore',
                 'Requests for Help/Needs': 'donation',
                 'Praying': 'not informative',
                 'Missing, trapped, or found people': 'casaulties and damage',
                 'Not Relevant': 'not informative',
                 'Informative': 'unlabeled',
                 'Injured or dead people': 'casaulties and damage',
                 'Infrastructure damage': 'casaulties and damage',
                 'Urgent Needs': 'donation',
                 'Not Informative': 'not informative',
                 'Not informative': 'not informative',
                 'Personal updates, sympathy, support': 'not informative',
                 'Other Relevant Information': 'unlabeled',
                 'Not relevant': 'not informative',
                 'Response Efforts': 'casaulties and damage',
                 'People missing or found': 'casaulties and damage',
                 'Humanitarian Aid Provided': 'donation',
                 'Money': 'donation',
                 'Caution and advice': 'caution and advice',
                 'Infrastructure Damage': 'casaulties and damage',
                 'Response efforts': 'casaulties and damage',
                 'Traditional media': 'ignore',
                 'Other useful information': 'unlabeled',
                 'No': 'ignore'}

    for root, dirs, files in os.walk(directory, topdown=False):
        for file in files:
            filename = os.path.join(root, file)

            if len(filename) > 4:
                if filename[-4:] == '.tsv' or filename[-4:] == '.csv':

                    print(filename)

                    for key in disaster_keys:
                        if key.lower() in filename.lower():
                            disaster_key = key

                    if "mers" in filename.lower():
                        disaster_key = "Middle_East_Respiratory_Syndrome"

                    encoding = 'ISO 8859-1'

                    with codecs.open(filename, 'r', encoding=encoding) as csvfile:

                        if filename[-3:] == 'tsv':
                            csv_reader = csv.reader(csvfile, delimiter='\t')
                        else:
                            csv_reader = csv.reader(csvfile)

                        for i, row in enumerate(csv_reader):
                            if i > 0 and len(row) >= 10:

                                simple_label = label_map[row[9]]

                                if simple_label != 'ignore':
                                    data[disaster_key]["tweet_ids"].append(row[0])
                                    tweet = str(row[7])
                                    tweet = tweet.encode("ascii", errors="ignore").decode()
                                    if len(tweet.split(" ")) > 300:
                                        print("Crowdflower2: "+" ".join(tweet))
                                    data[disaster_key]["tweets"].append(tweet)
                                    data[disaster_key]["labels"].append(row[9])
                                    data[disaster_key]["simple_labels"].append(simple_label)
                                    data[disaster_key]["source"].append("CrisisNLP_Volunteers")

                                    """

                                    if row[9] == 'Traditional media':
                                        print(tweet)
                                        print("\n\n")
                                    """

                                    # labels.append(row[9])

    # print(set(labels))

    #display_step = 100

    with open(data_dir, 'w') as outfile:
        json.dump(data, outfile)

    with open(data_dir_public, 'w') as outfile:
        for disaster_key in data:
            del data[disaster_key]["tweets"]
        json.dump(data, outfile)
