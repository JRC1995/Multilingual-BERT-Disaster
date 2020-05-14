import json
import re
from nltk.tokenize import TweetTokenizer
from wordsegment import load, segment

tw = TweetTokenizer()
load()
data_dir = "../Processed_Data/Processed_Data_Intermediate_Sentiment.json"
with open(data_dir) as file:
    data = json.load(file)

all_tweet_ids = []
all_tweets = []
all_labels = []

unique_tweets = {}


def split_hashtag(hashtag):
    phrase = hashtag[1:]
    if phrase.lower() in ["hagupit", "awaran", "rubyph",
                          "bhopa", "marinph", "bopha",
                          "rescueph", "pabloph", "reliefph",
                          "kkf", "floodph", "yolandaph",
                          "damayan", "yovanielrd", "bantanyan",
                          "midufinga", "jordsklv"]:
        segmented_words = [phrase]
    elif phrase.lower() == "typhoonhagupit":
        segmented_words = [phrase[0:7], phrase[7:]]
    else:
        segmented_words = segment(phrase)
    return segmented_words


def process_tweet(tweet):
    # pattern to match
    user_pattern = r'@[\w\.-]+[,\.;: ]+'
    # replace the matched pattern from string with,
    user_replace = r'<USER> '

    tweet = re.sub(user_pattern, user_replace, tweet)

    if tweet[0:3].lower() == "rt ":
        tweet = tweet[3:]

    tweet = tweet.split(" ")
    new_tweet = []
    for token in tweet:
        if 'https://' in token or 'http://' in token or 'www.' in token:
            new_tweet.append("<URL>")
        else:
            new_tweet.append(token)

    tweet = " ".join(new_tweet)

    tokenized_tweet = tw.tokenize(tweet)

    new_tokenized_tweet = []

    for token in tokenized_tweet:
        if '#' in token and len(token) > 1:
            new_tokenized_tweet.append('<HASH>')
            tokens = split_hashtag(token)
            new_tokenized_tweet += tokens
            new_tokenized_tweet.append('</HASH>')
        else:
            new_tokenized_tweet.append(token)

    tweet = " ".join(new_tokenized_tweet)

    return tweet


id = 0
tweet_ids = data["tweet_ids"]
tweets = data["tweets"]
labels = data["labels"]

for tweet_id, tweet, label in zip(tweet_ids, tweets, labels):
    tweet = process_tweet(tweet)
    tweet_id = str(tweet_id).strip("'")
    if tweet.strip() != '':
        if tweet not in unique_tweets:
            unique_tweets[tweet] = id
            all_tweet_ids.append(tweet_id)
            all_tweets.append(tweet)
            all_labels.append(label)

    if id % 10000 == 0:
        print("{} data processed ...".format(id))

    id += 1


d = {}
d["tweet_ids"] = all_tweet_ids
d["tweets"] = all_tweets
d["labels"] = all_labels

with open("../Processed_Data/sentiment.json", 'w') as outfile:
    json.dump(d, outfile)
