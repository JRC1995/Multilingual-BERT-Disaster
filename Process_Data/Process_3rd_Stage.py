from collections import Counter
from nltk.tokenize import TweetTokenizer
import json
import re
from langdetect import detect
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import numpy as np
import random

label_rank = {'casaulties and damage': 0,
              'caution and advice': 1,
              'donation': 2,
              'not informative': 3,
              'unlabeled': 4}


data_dir = "../Processed_Data/Processed_Data_Intermediate_Stage_2.json"
with open(data_dir) as file:
    data = json.load(file)

tweet_ids = data['tweet_ids']
tweets = data['tweets']
all_labels = data['labels']
all_disasters = data['disasters']

new_tweet_ids = []
new_tweets = []
new_labels = []
new_all_disasters = []
languages = []

i = 0
for tweet_id, tweet, labels, disaster_keys in zip(tweet_ids, tweets, all_labels, all_disasters):
    labels = list(set(labels))
    label_scores = [label_rank[label] for label in labels]
    label = labels[label_scores.index(min(label_scores))]

    try:
        language = detect(tweet)

        new_tweet_ids.append(tweet_id)
        new_tweets.append(tweet)
        new_labels.append(label)
        languages.append(language)
        new_all_disasters.append(disaster_keys)

    except:
        print("Can't detect language")

    i += 1

    if i % 10000 == 0:
        print("{} samples processed...".format(i))


# Analytics:

disaster_count = {}
label_count = {}
lang_count = {}
data_count = 0

for label, disaster_keys, lang in zip(new_labels, new_all_disasters, languages):

    data_count += 1

    lang_count[lang] = lang_count.get(lang, 0)+1
    label_count[label] = label_count.get(label, 0)+1

    for key in disaster_keys:
        disaster_count[key] = disaster_count.get(key, 0)+1


def color_func(word, font_size, position, orientation, random_state=None,
               **kwargs):
    return "hsl({}, {}%, {}%)".format(random.randint(116, 283), random.randint(60, 80), random.randint(35, 44))


wc = WordCloud(width=1920, height=1080, max_words=200,
               color_func=color_func, background_color='white')

# print(words)
word_cloud_dict = {}

for key in disaster_count:
    name = key.split("_")
    name = " ".join(name)
    word_cloud_dict[name] = disaster_count[key]

# Create and generate a word cloud image:
wordcloud = wc.generate_from_frequencies(word_cloud_dict)

# Display the generated image:
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud)
plt.axis("off")
# plt.show()
plt.savefig('stats/disastercloud.png', bbox_inches='tight')
plt.close()


file_data = "Total Data: {}".format(data_count)

file_data += "\n\nDisaster Statistics\n\n"

for key in disaster_count:
    name = key.split("_")
    name = " ".join(name)
    file_data += "{}: {}".format(name, disaster_count[key])+"\n"

file_data += "\n\nLanguage Statistics\n\n"

for lang in lang_count:
    file_data += "{}: {}".format(lang, lang_count[lang]) + "\n"

file_data += "\n\nLabel Statistics\n\n"

for label in label_count:
    file_data += "{}: {}".format(label, label_count[label]) + "\n"


f = open("stats/final_stats.txt", "w")
f.write(file_data)
f.close()

max_label_counts = max([v for k, v in label_count.items() if k != 'unlabeled'])
label_weights = {k: max_label_counts/v for k, v in label_count.items() if k != 'unlabeled'}
label_weights['informative'] = 1.0

with open("../Processed_Data/label_info.json", 'w') as outfile:
    json.dump(label_weights, outfile)

d = {}
d["tweet_ids"] = new_tweet_ids
d["tweets"] = new_tweets
d["labels"] = new_labels
d["languages"] = languages
d['disasters'] = new_all_disasters

with open("../Processed_Data/Processed_Data_Intermediate_Stage_3.json", 'w') as outfile:
    json.dump(d, outfile)
