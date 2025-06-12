##############################################################################
# LABEL CLUSTERS CHAINS FOR MULTILABEL CLASSIFICATION                        #
# Copyright (C) 2025                                                         #
#                                                                            #
# This code is free software: you can redistribute it and/or modify it under #
# the terms of the GNU General Public License as published by the Free       #
# Software Foundation, either version 3 of the License, or (at your option)  #
# any later version. This code is distributed in the hope that it will be    #
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of     #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General   #
# Public License for more details.                                           #
#                                                                            #
# Prof. Elaine Cecilia Gatto - UFLA - Lavras, Minas Gerais, Brazil           #
# Prof. Ricardo Cerri - USP - São Carlos, São Paulo, Brazil                  #
# Prof. Mauri Ferrandin - UFSC - Blumenau, Santa Catarina, Brazil            #
# Prof. Celine Vens - Ku Leuven - Kortrijik, West Flanders, Belgium          #
# PhD Felipe Nakano Kenji - Ku Leuven - Kortrijik, West Flanders, Belgium    #
#                                                                            #
# BIOMAL - http://www.biomal.ufscar.br                                       #
#                                                                            #
##############################################################################


import sys
import platform
import os

system = platform.system()
if system == 'Windows':
    user_profile = os.environ['USERPROFILE']
    FolderRoot = os.path.join(user_profile, 'Documents', 'MultiLabelEvaluationMetrics', 'src')
elif system in ['Linux', 'Darwin']:  # 'Darwin' is the system name for macOS
    FolderRoot = os.path.expanduser('~/LCCML/src')
else:
    raise Exception('Unsupported operating system')

os.chdir(FolderRoot)
current_directory = os.getcwd()
sys.path.append('..')

print("Caminhos em sys.path:")
for p in sys.path:
    print(p)


import pandas as pd
from sklearn.ensemble import RandomForestClassifier  

import importlib
import lccml
importlib.reload(lccml)
from lccml import LCCML

import evaluation as eval
importlib.reload(eval)

import measures as ms
importlib.reload(ms)


if __name__ == '__main__':    
    
    n_chains = 10
    random_state = 0
    n_estimators = 200
    baseModel = RandomForestClassifier(n_estimators = n_estimators, 
                                       random_state = random_state)

    train = pd.read_csv(sys.argv[1])
    valid = pd.read_csv(sys.argv[2])
    test = pd.read_csv(sys.argv[3])
    partitions = pd.read_csv(sys.argv[4])
    directory = sys.argv[5]       
    
    #train = pd.read_csv("/tmp/emotions/Datasets/emotions/CrossValidation/Tr/emotions-Split-Tr-1.csv")
    #test = pd.read_csv("/tmp/emotions/Datasets/emotions/CrossValidation/Ts/emotions-Split-Ts-1.csv")
    #valid = pd.read_csv("/tmp/emotions/Datasets/emotions/CrossValidation/Vl/emotions-Split-Vl-1.csv")
    #partitions = pd.read_csv("/tmp/emotions/Partitions/emotions/Split-1/Partition-2/partition-2.csv")
    #directory = "/tmp/emotions/Tested/Split-1"    

    train = pd.concat([train,valid],axis=0).reset_index(drop=True)

    clusters = partitions.groupby("group")["label"].apply(list)   
    allLabels = partitions["label"].unique()

    x_train = train.drop(allLabels, axis=1)
    y_train = train[allLabels]
    
    x_test = test.drop(allLabels, axis=1)
    y_test = test[allLabels]

    lccml = LCCML(baseModel,n_chains)
    lccml.fit(x_train,y_train,clusters,) 

    test_predictions = pd.DataFrame(lccml.predict(x_test))
    train_predictions = pd.DataFrame(lccml.predict(x_train))
    
    # test_predictions.columns
    # allLabels
    
    true = (directory + "/y_true.csv")
    pred = (directory + "/y_proba.csv")   
    train = (directory + "/y_train.csv") 
    
    test_predictions.to_csv(pred, index=False)
    test[allLabels].to_csv(true, index=False)

    train_predictions.to_csv(train, index=False)
    train_predictions[allLabels].to_csv(true, index=False)

    y_test.to_csv(true, index=False)     
    y_test[allLabels].to_csv(true, index=False)
    
    res_curves = eval.multilabel_curves_measures(y_test, test_predictions)
    res_lp = eval.multilabel_label_problem_measures(y_test, test_predictions)
    
    res_final = pd.concat([res_curves, res_lp], ignore_index=True)    
    name = (directory + "/results-python.csv") 
    res_final.to_csv(name, index=False)
  
    



