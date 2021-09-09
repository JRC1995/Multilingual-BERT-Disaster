# Credits
The [BERT-encoder codes](https://github.com/JRC1995/Multilingual-BERT-Disaster/tree/master/Lib/Encoders) are adapted and modified from Huggingface Transformers. 

# Multilingual-BERT-Disaster
Resources for: Cross-Lingual Disaster-related Multi-label Tweet Classification with Manifold Mixup (ACL SRW 2020)
https://www.aclweb.org/anthology/2020.acl-srw.39.pdf

See Processed_Data/ for anonymized (tweet_ids and labels only) splits. Test json files are different from validation or training, as it has data for different disasters separated out as in the paper. 

To get the tweet content, the best way is to download the appropriate files from crisisnlp and crisislex. Then, run ```bash process.sh``` in Process_Data/
The directory tree in the data folder should be like this: 

```
├── Data
│   ├── ACL_ICWSM_2018_datasets
│   │   ├── ACL_ICWSM_2018_datasets
│   │   │   ├── nepal
│   │   │   │   ├── 2015_Nepal_Earthquake_dev.tsv
│   │   │   │   ├── 2015_Nepal_Earthquake_test.tsv
│   │   │   │   ├── 2015_Nepal_Earthquake_train.tsv
│   │   │   │   └── 2015_Nepal_Earthquake_unlabelled_ids.txt
│   │   │   ├── queensland
│   │   │   │   ├── 2013_Queensland_Floods_dev.tsv
│   │   │   │   ├── 2013_Queensland_Floods_test.tsv
│   │   │   │   ├── 2013_Queensland_Floods_train.tsv
│   │   │   │   └── 2013_Queensland_Floods_unlabelled_ids.txt
│   │   │   └── README.txt
│   │   └── __MACOSX
│   │       └── ACL_ICWSM_2018_datasets
│   │           ├── nepal
│   │           └── queensland
│   ├── CrisisLexT26-v1.0
│   │   ├── CrisisLexT26
│   │   │   ├── 2012_Colorado_wildfires
│   │   │   │   ├── 2012_Colorado_wildfires-event_description.json
│   │   │   │   ├── 2012_Colorado_wildfires-tweetids_entire_period.csv
│   │   │   │   ├── 2012_Colorado_wildfires-tweets_labeled.csv
│   │   │   │   └── README.md
│   │   │   ├── 2012_Costa_Rica_earthquake
│   │   │   │   ├── 2012_Costa_Rica_earthquake-event_description.json
│   │   │   │   ├── 2012_Costa_Rica_earthquake-tweetids_entire_period.csv
│   │   │   │   ├── 2012_Costa_Rica_earthquake-tweets_labeled.csv
│   │   │   │   └── README.md
│   │   │   ├── 2012_Guatemala_earthquake
│   │   │   │   ├── 2012_Guatemala_earthquake-event_description.json
│   │   │   │   ├── 2012_Guatemala_earthquake-tweetids_entire_period.csv
│   │   │   │   ├── 2012_Guatemala_earthquake-tweets_labeled.csv
│   │   │   │   └── README.md
│   │   │   ├── 2012_Italy_earthquakes
│   │   │   │   ├── 2012_Italy_earthquakes-event_description.json
│   │   │   │   ├── 2012_Italy_earthquakes-tweetids_entire_period.csv
│   │   │   │   ├── 2012_Italy_earthquakes-tweets_labeled.csv
│   │   │   │   └── README.md
│   │   │   ├── 2012_Philipinnes_floods
│   │   │   │   ├── 2012_Philipinnes_floods-event_description.json
│   │   │   │   ├── 2012_Philipinnes_floods-tweetids_entire_period.csv
│   │   │   │   ├── 2012_Philipinnes_floods-tweets_labeled.csv
│   │   │   │   └── README.md
│   │   │   ├── 2012_Typhoon_Pablo
│   │   │   │   ├── 2012_Typhoon_Pablo-event_description.json
│   │   │   │   ├── 2012_Typhoon_Pablo-tweetids_entire_period.csv
│   │   │   │   ├── 2012_Typhoon_Pablo-tweets_labeled.csv
│   │   │   │   └── README.md
│   │   │   ├── 2012_Venezuela_refinery
│   │   │   │   ├── 2012_Venezuela_refinery-event_description.json
│   │   │   │   ├── 2012_Venezuela_refinery-tweetids_entire_period.csv
│   │   │   │   ├── 2012_Venezuela_refinery-tweets_labeled.csv
│   │   │   │   └── README.md
│   │   │   ├── 2013_Alberta_floods
│   │   │   │   ├── 2013_Alberta_floods-event_description.json
│   │   │   │   ├── 2013_Alberta_floods-tweetids_entire_period.csv
│   │   │   │   ├── 2013_Alberta_floods-tweets_labeled.csv
│   │   │   │   └── README.md
│   │   │   ├── 2013_Australia_bushfire
│   │   │   │   ├── 2013_Australia_bushfire-event_description.json
│   │   │   │   ├── 2013_Australia_bushfire-tweetids_entire_period.csv
│   │   │   │   ├── 2013_Australia_bushfire-tweets_labeled.csv
│   │   │   │   └── README.md
│   │   │   ├── 2013_Bohol_earthquake
│   │   │   │   ├── 2013_Bohol_earthquake-event_description.json
│   │   │   │   ├── 2013_Bohol_earthquake-tweetids_entire_period.csv
│   │   │   │   ├── 2013_Bohol_earthquake-tweets_labeled.csv
│   │   │   │   └── README.md
│   │   │   ├── 2013_Boston_bombings
│   │   │   │   ├── 2013_Boston_bombings-event_description.json
│   │   │   │   ├── 2013_Boston_bombings-tweetids_entire_period.csv
│   │   │   │   ├── 2013_Boston_bombings-tweets_labeled.csv
│   │   │   │   └── README.md
│   │   │   ├── 2013_Brazil_nightclub_fire
│   │   │   │   ├── 2013_Brazil_nightclub_fire-event_description.json
│   │   │   │   ├── 2013_Brazil_nightclub_fire-tweetids_entire_period.csv
│   │   │   │   ├── 2013_Brazil_nightclub_fire-tweets_labeled.csv
│   │   │   │   └── README.md
│   │   │   ├── 2013_Colorado_floods
│   │   │   │   ├── 2013_Colorado_floods-event_description.json
│   │   │   │   ├── 2013_Colorado_floods-tweetids_entire_period.csv
│   │   │   │   ├── 2013_Colorado_floods-tweets_labeled.csv
│   │   │   │   └── README.md
│   │   │   ├── 2013_Glasgow_helicopter_crash
│   │   │   │   ├── 2013_Glasgow_helicopter_crash-event_description.json
│   │   │   │   ├── 2013_Glasgow_helicopter_crash-tweetids_entire_period.csv
│   │   │   │   ├── 2013_Glasgow_helicopter_crash-tweets_labeled.csv
│   │   │   │   └── README.md
│   │   │   ├── 2013_LA_airport_shootings
│   │   │   │   ├── 2013_LA_airport_shootings-event_description.json
│   │   │   │   ├── 2013_LA_airport_shootings-tweetids_entire_period.csv
│   │   │   │   ├── 2013_LA_airport_shootings-tweets_labeled.csv
│   │   │   │   └── README.md
│   │   │   ├── 2013_Lac_Megantic_train_crash
│   │   │   │   ├── 2013_Lac_Megantic_train_crash-event_description.json
│   │   │   │   ├── 2013_Lac_Megantic_train_crash-tweetids_entire_period.csv
│   │   │   │   ├── 2013_Lac_Megantic_train_crash-tweets_labeled.csv
│   │   │   │   └── README.md
│   │   │   ├── 2013_Manila_floods
│   │   │   │   ├── 2013_Manila_floods-event_description.json
│   │   │   │   ├── 2013_Manila_floods-tweetids_entire_period.csv
│   │   │   │   ├── 2013_Manila_floods-tweets_labeled.csv
│   │   │   │   └── README.md
│   │   │   ├── 2013_NY_train_crash
│   │   │   │   ├── 2013_NY_train_crash-event_description.json
│   │   │   │   ├── 2013_NY_train_crash-tweetids_entire_period.csv
│   │   │   │   ├── 2013_NY_train_crash-tweets_labeled.csv
│   │   │   │   └── README.md
│   │   │   ├── 2013_Queensland_floods
│   │   │   │   ├── 2013_Queensland_floods-event_description.json
│   │   │   │   ├── 2013_Queensland_floods-tweetids_entire_period.csv
│   │   │   │   ├── 2013_Queensland_floods-tweets_labeled.csv
│   │   │   │   └── README.md
│   │   │   ├── 2013_Russia_meteor
│   │   │   │   ├── 2013_Russia_meteor-event_description.json
│   │   │   │   ├── 2013_Russia_meteor-tweetids_entire_period.csv
│   │   │   │   ├── 2013_Russia_meteor-tweets_labeled.csv
│   │   │   │   └── README.md
│   │   │   ├── 2013_Sardinia_floods
│   │   │   │   ├── 2013_Sardinia_floods-event_description.json
│   │   │   │   ├── 2013_Sardinia_floods-tweetids_entire_period.csv
│   │   │   │   ├── 2013_Sardinia_floods-tweets_labeled.csv
│   │   │   │   └── README.md
│   │   │   ├── 2013_Savar_building_collapse
│   │   │   │   ├── 2013_Savar_building_collapse-event_description.json
│   │   │   │   ├── 2013_Savar_building_collapse-tweetids_entire_period.csv
│   │   │   │   ├── 2013_Savar_building_collapse-tweets_labeled.csv
│   │   │   │   └── README.md
│   │   │   ├── 2013_Singapore_haze
│   │   │   │   ├── 2013_Singapore_haze-event_description.json
│   │   │   │   ├── 2013_Singapore_haze-tweetids_entire_period.csv
│   │   │   │   ├── 2013_Singapore_haze-tweets_labeled.csv
│   │   │   │   └── README.md
│   │   │   ├── 2013_Spain_train_crash
│   │   │   │   ├── 2013_Spain_train_crash-event_description.json
│   │   │   │   ├── 2013_Spain_train_crash-tweetids_entire_period.csv
│   │   │   │   ├── 2013_Spain_train_crash-tweets_labeled.csv
│   │   │   │   └── README.md
│   │   │   ├── 2013_Typhoon_Yolanda
│   │   │   │   ├── 2013_Typhoon_Yolanda-event_description.json
│   │   │   │   ├── 2013_Typhoon_Yolanda-tweetids_entire_period.csv
│   │   │   │   ├── 2013_Typhoon_Yolanda-tweets_labeled.csv
│   │   │   │   └── README.md
│   │   │   ├── 2013_West_Texas_explosion
│   │   │   │   ├── 2013_West_Texas_explosion-event_description.json
│   │   │   │   ├── 2013_West_Texas_explosion-tweetids_entire_period.csv
│   │   │   │   ├── 2013_West_Texas_explosion-tweets_labeled.csv
│   │   │   │   └── README.md
│   │   │   └── README.md
│   │   └── __MACOSX
│   │       └── CrisisLexT26
│   ├── CrisisLexT6
│   │   ├── 2012_Sandy_Hurricane
│   │   │   └── 2012_Sandy_Hurricane-ontopic_offtopic.csv
│   │   ├── 2013_Alberta_Floods
│   │   │   └── 2013_Alberta_Floods-ontopic_offtopic.csv
│   │   ├── 2013_Boston_Bombings
│   │   │   └── 2013_Boston_Bombings-ontopic_offtopic.csv
│   │   ├── 2013_Oklahoma_Tornado
│   │   │   └── 2013_Oklahoma_Tornado-ontopic_offtopic.csv
│   │   ├── 2013_Queensland_Floods
│   │   │   └── 2013_Queensland_Floods-ontopic_offtopic.csv
│   │   ├── 2013_West_Texas_Explosion
│   │   │   └── 2013_West_Texas_Explosion-ontopic_offtopic.csv
│   │   └── README.md
│   ├── CrisisMMD_v1.0
│   │   ├── annotations
│   │   │   ├── california_wildfires_final_data.tsv
│   │   │   ├── hurricane_harvey_final_data.tsv
│   │   │   ├── hurricane_irma_final_data.tsv
│   │   │   ├── hurricane_maria_final_data.tsv
│   │   │   ├── iraq_iran_earthquake_final_data.tsv
│   │   │   ├── mexico_earthquake_final_data.tsv
│   │   │   └── srilanka_floods_final_data.tsv
│   │   ├── json
│   │   │   ├── california_wildfires_final_data.json
│   │   │   ├── hurricane_harvey_final_data.json
│   │   │   ├── hurricane_irma_final_data.json
│   │   │   ├── hurricane_maria_final_data.json
│   │   │   ├── iraq_earthquake_final_data.json
│   │   │   ├── mexico_earthquake_final_data.json
│   │   │   └── srilanka_floods_final_data.json
│   │   └── Readme.txt
│   ├── CrisisNLP_labeled_data_crowdflower_v2
│   │   ├── CrisisNLP_labeled_data_crowdflower
│   │   │   ├── 2013_Pakistan_eq
│   │   │   │   ├── 2013_Pakistan_eq_CF_labeled_data.tsv
│   │   │   │   └── labeling-instructions.txt
│   │   │   ├── 2014_California_Earthquake
│   │   │   │   ├── 2014_California_Earthquake_CF_labeled_data.tsv
│   │   │   │   └── labeling-instructions.txt
│   │   │   ├── 2014_Chile_Earthquake_cl
│   │   │   │   ├── 2014_Chile_Earthquake_cl_labeled_data.tsv
│   │   │   │   └── labeling-instructions.txt
│   │   │   ├── 2014_Chile_Earthquake_en
│   │   │   │   ├── 2014_Chile_Earthquake_en_CF_labeled_data.tsv
│   │   │   │   └── labeling-instructions.txt
│   │   │   ├── 2014_ebola_cf
│   │   │   │   ├── 2014_ebola_CF_labeled_data.tsv
│   │   │   │   └── labeling-instructions.txt
│   │   │   ├── 2014_Hurricane_Odile_Mexico_en
│   │   │   │   ├── 2014_Hurricane_Odile_Mexico_en_CF_labeled_data.tsv
│   │   │   │   └── labeling-instructions.txt
│   │   │   ├── 2014_India_floods
│   │   │   │   ├── 2014_India_floods_CF_labeled_data.tsv
│   │   │   │   └── labeling-instructions.txt
│   │   │   ├── 2014_Middle_East_Respiratory_Syndrome_en
│   │   │   │   ├── 2014_MERS_en_CF_labeled_data.tsv
│   │   │   │   └── labeling-instructions.txt
│   │   │   ├── 2014_Pakistan_floods
│   │   │   │   ├── 2014_Pakistan_floods_CF_labeled_data.tsv
│   │   │   │   └── labeling-instructions.txt
│   │   │   ├── 2014_Philippines_Typhoon_Hagupit_en
│   │   │   │   ├── 2014_Philippines_Typhoon_Hagupit_en_CF_labeled_data.tsv
│   │   │   │   └── labeling-instructions.txt
│   │   │   ├── 2015_Cyclone_Pam_en
│   │   │   │   ├── 2015_Cyclone_Pam_en_CF_labeled_data.tsv
│   │   │   │   └── labeling-instructions.txt
│   │   │   ├── 2015_Nepal_Earthquake_en
│   │   │   │   ├── 2015_Nepal_Earthquake_en_CF_labeled_data.tsv
│   │   │   │   └── labeling-instructions.txt
│   │   │   ├── README.txt
│   │   │   └── Terms of use.txt
│   │   └── __MACOSX
│   │       └── CrisisNLP_labeled_data_crowdflower
│   │           ├── 2013_Pakistan_eq
│   │           ├── 2014_California_Earthquake
│   │           ├── 2014_Chile_Earthquake_cl
│   │           ├── 2014_Chile_Earthquake_en
│   │           ├── 2014_ebola_cf
│   │           ├── 2014_Hurricane_Odile_Mexico_en
│   │           ├── 2014_India_floods
│   │           ├── 2014_Middle_East_Respiratory_Syndrome_en
│   │           ├── 2014_Pakistan_floods
│   │           ├── 2014_Philippines_Typhoon_Hagupit_en
│   │           ├── 2015_Cyclone_Pam_en
│   │           └── 2015_Nepal_Earthquake_en
│   ├── CrisisNLP_volunteers_labeled_data
│   │   ├── CrisisNLP_volunteers_labeled_data
│   │   │   ├── 2014_California_Earthquake
│   │   │   │   ├── 2014_California_Earthquake.csv
│   │   │   │   └── labeling-instructions.txt
│   │   │   ├── 2014_Chile_Earthquake_cl
│   │   │   │   ├── 2014_chile_earthquake_cl.csv
│   │   │   │   └── labeling-instructions.txt
│   │   │   ├── 2014_Chile_Earthquake_en
│   │   │   │   ├── 2014_Chile_Earthquake_en.csv
│   │   │   │   └── labeling-instructions.txt
│   │   │   ├── 2014_Hurricane_Odile_Mexico_en
│   │   │   │   ├── 2014_Hurricane_Odile_Mexico_en.csv
│   │   │   │   └── labeling-instructions.txt
│   │   │   ├── 2014_Iceland_Volcano_en
│   │   │   │   ├── 2014_Iceland_Volcano_en.csv
│   │   │   │   └── labeling-instructions.txt
│   │   │   ├── 2014_Malaysia_Airline_MH370_en
│   │   │   │   ├── 2014_Malaysia_Airline_MH370_en.csv
│   │   │   │   └── labeling-instructions.txt
│   │   │   ├── 2014_Middle_East_Respiratory_Syndrome_en
│   │   │   │   ├── 2014_Middle_East_Respiratory_Syndrome_en.csv
│   │   │   │   └── labeling-instructions.txt
│   │   │   ├── 2014_Philippines_Typhoon_Hagupit_en
│   │   │   │   ├── 2014_Typhoon_Hagupit_en.csv
│   │   │   │   └── labeling-instructions.txt
│   │   │   ├── 2015_Cyclone_Pam_en
│   │   │   │   ├── 2015_Cyclone_Pam_en.csv
│   │   │   │   └── labeling-instructions.txt
│   │   │   ├── 2015_Nepal_Earthquake_en
│   │   │   │   ├── 2015_Nepal_Earthquake_en.csv
│   │   │   │   └── labeling-instructions.txt
│   │   │   ├── Landslides_Worldwide_en
│   │   │   │   ├── labeling-instructions.txt
│   │   │   │   └── Landslides_Worldwide_en.csv
│   │   │   ├── Landslides_Worldwide_esp
│   │   │   │   ├── labeling-instructions.txt
│   │   │   │   └── Landslides_Worldwide_esp.csv
│   │   │   ├── Landslides_Worldwide_fr
│   │   │   │   ├── labeling-instructions.txt
│   │   │   │   └── LandSlides_Worldwide_fr.csv
│   │   │   ├── README.txt
│   │   │   └── Terms of use.txt
│   │   └── __MACOSX
│   │       └── CrisisNLP_volunteers_labeled_data
│   │           ├── 2014_California_Earthquake
│   │           ├── 2014_Chile_Earthquake_cl
│   │           ├── 2014_Chile_Earthquake_en
│   │           ├── 2014_Hurricane_Odile_Mexico_en
│   │           ├── 2014_Iceland_Volcano_en
│   │           ├── 2014_Malaysia_Airline_MH370_en
│   │           ├── 2014_Middle_East_Respiratory_Syndrome_en
│   │           ├── 2014_Philippines_Typhoon_Hagupit_en
│   │           ├── 2015_Cyclone_Pam_en
│   │           ├── 2015_Nepal_Earthquake_en
│   │           ├── Landslides_Worldwide_en
│   │           ├── Landslides_Worldwide_esp
│   │           └── Landslides_Worldwide_fr
│   └── deep-learning-for-big-crisis-data-master
│       ├── data
│       │   ├── nn_data
│       │   │   ├── dev.csv -> ../sample_prccd_dev.csv
│       │   │   ├── test.csv -> ../sample_prccd_test.csv
│       │   │   └── train.csv -> ../sample_prccd_train.csv
│       │   ├── sample.csv
│       │   ├── sample_prccd.csv
│       │   ├── sample_prccd_dev.csv
│       │   ├── sample_prccd_test.csv
│       │   └── sample_prccd_train.csv
│       ├── data_helpers
│       │   ├── preprocess.py
│       │   ├── split_data.py
│       │   ├── twokenize.py
│       │   └── twokenize.pyc
│       ├── dnn_scripts
│       │   ├── cnn_crisis.py
│       │   └── utilities
│       │       ├── aidr.py
│       │       ├── aidr.pyc
│       │       ├── filter_emb.py
│       │       ├── __init__.py
│       │       └── __init__.pyc
│       ├── embeddings
│       │   └── crisis_embeddings.text
│       ├── README.md
│       └── run_cnn.sh

```
You can get the directory structure like this simply by downloading and extracting files from crisisnlp and crisislex

```
To prepare our large multilingual dataset, we aggregated several resources from CrisisNLP,2
together with two resources from CrisisLex.3 Specifically, we used Resource #1 (Imran et al., 2016a),
Resource #4 (Nguyen et al., 2017), Resource #5
(Alam et al., 2018c), and Resource #7 (Alam et al.,
2018a) from CrisisNLP, and CrisisLexT6 (Olteanu
et al., 2014) and CrisisLexT26 (Olteanu et al.,
2015) from CrisisLex.
```

# Instructions
1. Run `save_pre_trained_locally.py` to download pre-trained weights.
2. Download [crawl-300d-2M-subword.zip](https://fasttext.cc/docs/en/english-vectors.html). Extract and put the extracted file `crawl-300d-2M-subword.bin` into `Embeddings/`. 
3. Download [glove.twitter.27B.zip](https://nlp.stanford.edu/projects/glove/). Extract and put the extracted file into `Embeddings/`.
4. Run `bash process.sh` from `Process_Data/` (if you haven't already) to preprocess the downloaded files.
5. To train, go to `train/` and run `train.py --model=X --lang=Y` where X can be any model you want to run in `["BOW_mean", "FastText", "BERT_word_mixup", "BERT_sent_mixup", "CNN", "XML_CNN", "DenseCNN", "BiLSTM", "BERT", "BERT_mixup]` (BERT_mixup is BERT + manifold mixup) and Y is `en` for using English only dataset and `all` for multilingual. 
6. To test, go to `test/` and run `test.py --model=X --lang=Y` where X and Y means the same thing as in 3. 

Email me for any question or other resources.

# Citation:

```
@inproceedings{ray-chowdhury-etal-2020-cross,
    title = "Cross-Lingual Disaster-related Multi-label Tweet Classification with Manifold Mixup",
    author = "Ray Chowdhury, Jishnu  and
      Caragea, Cornelia  and
      Caragea, Doina",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-srw.39",
    doi = "10.18653/v1/2020.acl-srw.39",
    pages = "292--298",
    abstract = "Distinguishing informative and actionable messages from a social media platform like Twitter is critical for facilitating disaster management. For this purpose, we compile a multilingual dataset of over 130K samples for multi-label classification of disaster-related tweets. We present a masking-based loss function for partially labelled samples and demonstrate the effectiveness of Manifold Mixup in the text domain. Our main model is based on Multilingual BERT, which we further improve with Manifold Mixup. We show that our model generalizes to unseen disasters in the test set. Furthermore, we analyze the capability of our model for zero-shot generalization to new languages. Our code, dataset, and other resources are available on Github.",
}
```
