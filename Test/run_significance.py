#from testSignificance import mcNemar, calculateContingency
import numpy as np
from scipy import stats
from pathlib import Path


def calculateContingency(data_A, data_B, n):
    ABrr = 0
    ABrw = 0
    ABwr = 0
    ABww = 0
    for i in range(0, n):
        if(data_A[i] == 1 and data_B[i] == 1):
            ABrr = ABrr+1
        if (data_A[i] == 1 and data_B[i] == 0):
            ABrw = ABrw + 1
        if (data_A[i] == 0 and data_B[i] == 1):
            ABwr = ABwr + 1
        else:
            ABww = ABww + 1
    return np.array([[ABrr, ABrw], [ABwr, ABww]])


def mcNemar(table):
    statistic = float(np.abs(table[0][1]-table[1][0]))**2/(table[1][0]+table[0][1])
    pval = 1-stats.chi2.cdf(statistic, 1)
    return pval


alpha = 0.05

model_A = "BERT"
model_B = "BERT_sent_mixup"
language = "en"
times = [0, 1, 2, 3, 4]
disasters = ["Russia_Meteor", "Cyclone_Pam", "Philipinnes_Floods", "Mixed"]


for time in times:
    for disaster in disasters:
        filename_A = Path("output/{}_{}_{}_{}.txt".format(disaster,
                                                          model_A, time, language))
        filename_B = Path("output/{}_{}_{}_{}.txt".format(disaster,
                                                          model_B, time, language))

        with open(filename_A) as f:
            data_A = f.read().splitlines()

        with open(filename_B) as f:
            data_B = f.read().splitlines()

        data_A = list(map(float, data_A))
        data_B = list(map(float, data_B))

        # print(data_A)
        # print(data_B)

        #f_obs = calculateContingency(data_A, data_B, len(data_A))
        #pval = mcNemar(f_obs)

        results = stats.ttest_rel(data_A, data_B)
        pval = float(results[1]) / 2

        significant = (pval <= alpha)

        print("Model A: {}, Model B: {} | Disaster: {} | P-val: {} | Significant: {}".format(
            model_A, model_B, disaster, pval, significant))
    print("\n\n")
