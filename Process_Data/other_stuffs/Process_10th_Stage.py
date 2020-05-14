import json
import re
import random
from nltk.tokenize import TweetTokenizer
import pickle
import collections

random.seed(101)
tw = TweetTokenizer()

data_dir = "../Processed_Data/Processed_Data_Intermediate_Stage_3.json"
with open(data_dir) as file:
    data = json.load(file)

tweet_ids = data['tweet_ids']
tweets = data['tweets']
labels = data['labels']
languages = data['languages']
all_disasters = data['disasters']

# print(set(languages))

test_keys = ['Russia_Meteor', 'Cyclone_Pam',
             'Philipinnes_Floods', 'French', 'Italian', 'Spanish', 'Mixed']


test_tweet_ids = {k: [] for k in test_keys}
test_tweets = {k: [] for k in test_keys}
test_labels = {k: [] for k in test_keys}

spanish_tweet_ids = []
spanish_tweets = []
spanish_labels = []

rest_tweet_ids = []
rest_tweets = []
rest_labels = []

russia_languages = []

for tweet_id, tweet, label, language, disasters in zip(tweet_ids,
                                                       tweets,
                                                       labels,
                                                       languages,
                                                       all_disasters):

    flag = 0

    for key in test_keys:
        if key in disasters:
            if key == 'Russia_Meteor':
                russia_languages.append(language)
            test_tweet_ids[key].append(tweet_id)
            test_tweets[key].append(tweet)
            test_labels[key].append(label)
            flag = 1

    if flag == 0:
        lower_tweet = tweet.lower()
        key = 'none'
        if "russia" in lower_tweet and "meteor" in lower_tweet:
            key = 'Russia_Meteor'
        elif "cyclone" in lower_tweet and "pam" in lower_tweet:
            key = 'Cyclone_Pam'
        elif "philipinnes" in lower_tweet and "flood" in lower_tweet:
            key = 'Philipinnes_Floods'

        if key != 'none':
            if key == 'Russia_Meteor':
                russia_languages.append(language)
            test_tweet_ids[key].append(tweet_id)
            test_tweets[key].append(tweet)
            test_labels[key].append(label)
            flag = 1

    # if language != 'en':
        #print("Language: {}, Tweet: {}".format(language, tweet))

    if language == 'es':
        spanish_tweet_ids.append(tweet_id)
        spanish_tweets.append(tweet)
        spanish_labels.append(label)
        flag = 1
    elif language == 'fr':
        test_tweet_ids['French'].append(tweet_id)
        test_tweets['French'].append(tweet)
        test_labels['French'].append(label)
        flag = 1

    elif language == 'it':
        test_tweet_ids['Italian'].append(tweet_id)
        test_tweets['Italian'].append(tweet)
        test_labels['Italian'].append(label)
        flag = 1

    if flag == 0:
        rest_tweet_ids.append(tweet_id)
        rest_tweets.append(tweet)
        rest_labels.append(label)


russia_language_counter = {}

for language in russia_languages:
    russia_language_counter[language] = russia_language_counter.get(language, 0)+1

f = open("stats/russia_meteor_language_analysis.txt", "w")

file_data = ""

for language in russia_language_counter:
    file_data += language + ": " + str(russia_language_counter[language]) + "\n"
f.write(file_data)
f.close()

randomized_indices = [i for i in range(len(spanish_tweet_ids))]
random.shuffle(randomized_indices)

spanish_tweet_ids = [spanish_tweet_ids[i] for i in randomized_indices]
spanish_tweets = [spanish_tweets[i] for i in randomized_indices]
spanish_labels = [spanish_labels[i] for i in randomized_indices]

len_spanish = len(randomized_indices)

test_tweet_ids['Spanish'] = spanish_tweet_ids
test_tweets['Spanish'] = spanish_tweets
test_labels['Spanish'] = spanish_labels


randomized_indices = [i for i in range(len(rest_tweet_ids))]
random.shuffle(randomized_indices)

tweet_ids = [rest_tweet_ids[i] for i in randomized_indices]
tweets = [rest_tweets[i] for i in randomized_indices]
labels = [rest_labels[i] for i in randomized_indices]

test_len = 5000
val_len = 10000

test_tweet_ids['Mixed'] = tweet_ids[0: test_len]
test_tweets['Mixed'] = tweets[0: test_len]
test_labels['Mixed'] = labels[0: test_len]

val_tweet_ids = tweet_ids[test_len: val_len+test_len]
val_tweets = tweets[test_len: val_len+test_len]
val_labels = labels[test_len: val_len+test_len]

train_tweet_ids = tweet_ids[val_len+test_len:]
train_tweets = tweets[val_len+test_len:]
train_labels = labels[val_len+test_len:]


print("\n\nExample Training Data\n\n")
for i in range(10):
    print("ID:", train_tweet_ids[i])
    print("Tweet:", train_tweets[i])
    print("Label:", train_labels[i])
    print("\n")

print("\n\nExample Validation Data\n\n")
for i in range(10):
    print("ID:", val_tweet_ids[i])
    print("Tweet:", val_tweets[i])
    print("Label:", val_labels[i])
    print("\n")

print("\n\nExample Testing Data\n\n")

for key in test_tweet_ids:
    print("{}\n\n".format(key))
    for i in range(10):
        print("ID:", test_tweet_ids[key][i])
        print("Tweet:", test_tweets[key][i])
        print("Label:", test_labels[key][i])
        print("\n")

file_data = "Training Data: {}".format(len(train_tweet_ids))+"\n"
file_data += "Validation Data: {}".format(len(val_tweet_ids))+"\n"
test_len = sum([len(test_tweet_ids[key]) for key in test_tweet_ids])
file_data += "Testing Data: {}".format(test_len)+"\n"

f = open("stats/split_stats_analysis.txt", "w")
f.write(file_data)
f.close()

d = {}
d["tweet_ids"] = train_tweet_ids
d["tweets"] = train_tweets
d["labels"] = train_labels

with open("../Processed_Data/train_data_analysis.json", 'w') as outfile:
    json.dump(d, outfile)

d = {}
d["tweet_ids"] = val_tweet_ids
d["tweets"] = val_tweets
d["labels"] = val_labels

with open("../Processed_Data/val_data_analysis.json", 'w') as outfile:
    json.dump(d, outfile)

d = {}
d["tweet_ids"] = test_tweet_ids
d["tweets"] = test_tweets
d["labels"] = test_labels

with open("../Processed_Data/test_data_analysis.json", 'w') as outfile:
    json.dump(d, outfile)
