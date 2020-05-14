import subscripts.BigCrisisData
import subscripts.CrisisLexT6
import subscripts.CrisisLexT26
import subscripts.CrisisMMD
import subscripts.CrisisNLP_Crowdflower
import subscripts.CrisisNLP_Volunteers
import subscripts.ICWSM_2018
import json
import os

data_dir = "../Processed_Data/Processed_Data_Intermediate.json"
data_dir_public = "../Processed_Data/Processed_Data_Intermediate_Public.json"

if os.path.exists(data_dir):
    os.remove(data_dir)

if os.path.exists(data_dir_public):
    os.remove(data_dir_public)

subscripts.BigCrisisData.process(data_dir, data_dir_public)
subscripts.CrisisLexT6.process(data_dir, data_dir_public)
subscripts.CrisisLexT26.process(data_dir, data_dir_public)
subscripts.CrisisMMD.process(data_dir, data_dir_public)
subscripts.CrisisNLP_Crowdflower.process(data_dir, data_dir_public)
subscripts.CrisisNLP_Volunteers.process(data_dir, data_dir_public)
subscripts.ICWSM_2018.process(data_dir, data_dir_public)
