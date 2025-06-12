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


import joblib
import io
import csv
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.base import clone

class LCCML:
    """
    Label Clusters Chains for Multi-Label Classification (LCCML).

    The algorithm trains an ensemble of classifier chains, where each chain
    operates on a random permutation of label clusters. Each cluster can contain 
    one or multiple labels. The prediction of previous clusters is added as input 
    for subsequent clusters in the chain.

    Additional features implemented:
    - Measures training time per chain and total training time.
    - Measures model size per chain and total model size in memory (without disk I/O).
    - Measures total prediction time.

    Parameters
    ----------
    model : sklearn-like estimator
        Base classifier to be used for each cluster.
    
    n_chains : int, default=10
        Number of chains in the ensemble.

    Attributes
    ----------
    orders : list
        Stores the random order of clusters for each chain.
    
    chains : list
        Trained chains (list of models per chain).
    
    chain_train_times : list
        Training time for each chain.
    
    train_time_total : float
        Total training time.
    
    chain_model_sizes : list
        Size (in bytes) of each chain (in memory).
    
    total_model_size : int
        Total size (sum of all chains).
    
    test_time_total : float
        Accumulated total prediction time.

    Example of Usage
    ----------------
    from sklearn.ensemble import RandomForestClassifier

    model_base = RandomForestClassifier()
    
    lccml = LCCML(model=model_base, n_chains=10)

    clusters = [['label1', 'label2'], ['label3'], ['label4', 'label5']]

    lccml.fit(X_train, Y_train, clusters)
    
    Y_pred = lccml.predict(X_test)

    print("Total training time:", lccml.train_time_total)
    
    print("Training times per chain:", lccml.chain_train_times)
    
    print("Model sizes (bytes):", lccml.chain_model_sizes)
    
    print("Total model size (bytes):", lccml.total_model_size)
    
    print("Total prediction time:", lccml.test_time_total)
    """

    random_seed = 1234
    rng = np.random.default_rng(random_seed)


    # ----------------------------------------------- #
    #                                                 #
    # ----------------------------------------------- #
    def __init__(self, model, n_chains=10):             
        self.model = model          
        self.n_chains = n_chains    
        self.orders = None         
        self.chains = []           
        self.chain_train_times = []
        self.train_time_total = 0
        self.test_time_total = 0 
        self.chain_model_sizes = []
        self.total_model_size = 0


    # ----------------------------------------------- #
    #                                                 #
    # ----------------------------------------------- #
    def fit(self, x, y, clusters):
        """
        Train the ensemble of label clusters chains.

        This method trains multiple classifier chains. Each chain follows a random 
        permutation of the provided label clusters. The model uses the predictions 
        of previous clusters as additional features when training subsequent clusters.

        Parameters
        ----------
        x : pd.DataFrame
            Feature matrix of shape (n_samples, n_features).
        
        y : pd.DataFrame
            Multi-label target matrix of shape (n_samples, n_labels), where columns correspond to label names.
        
        clusters : list of list of str
            Each sublist contains the names of labels belonging to one cluster.
            Example: [['label1', 'label2'], ['label3'], ['label4', 'label5']].

        Returns
        -------
        None

        Example
        -------
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> lccml = LCCML(model=model, n_chains=5)

        >>> # Suppose X_train is a pandas DataFrame of features, and Y_train is a pandas DataFrame of labels
        >>> clusters = [['label_a', 'label_b'], ['label_c'], ['label_d', 'label_e']]
        >>> lccml.fit(X_train, Y_train, clusters)

        After fitting, you can access:
        >>> lccml.train_time_total  # total training time
        >>> lccml.chain_train_times # list of training times per chain
        >>> lccml.total_model_size  # total model size (bytes)
        """
        
        self.__generateOrders(len(clusters))
        self.clusters = self.__preprocessClustersName(clusters, y)        
        for i in range(self.n_chains):
            self.__fitChain(self.orders[i], x, y)
        self.train_time_total = sum(self.chain_train_times)
        self.total_model_size = sum(self.chain_model_sizes)
    
    
    # ----------------------------------------------- #
    #                                                 #
    # ----------------------------------------------- #
    def __generateOrders(self, n_clusters):
        """
        Generate random permutations of cluster orders for each chain.

        Each chain will follow a different random order of clusters, ensuring
        diversity across the ensemble of label clusters chains. The generated orders
        are stored in self.orders as a list of numpy arrays.

        Parameters
        ----------
        n_clusters : int
            Total number of clusters.

        Returns
        -------
        None

        Example
        -------
        >>> lccml = LCCML(model=RandomForestClassifier(), n_chains=3)
        >>> lccml.__generateOrders(n_clusters=4)
        >>> print(lccml.orders)
        [array([2, 1, 3, 0]), array([0, 2, 1, 3]), array([3, 0, 1, 2])]

        Each array represents one chain's cluster processing order.
        """
        self.orders = [self.rng.permutation(n_clusters) for _ in range(self.n_chains)]

    
    # ----------------------------------------------- #
    #                                                 #
    # ----------------------------------------------- #
    def __fitChain(self, order, x, y):
        """
        Train a single label cluster chain following the specified cluster order.

        For each cluster in the given order, the corresponding labels are trained 
        using the base model. After training on a cluster, its predicted labels are 
        appended as additional features for training the next cluster.

        The training time and model size of the chain are recorded.

        Parameters
        ----------
        order : array-like of int
            A permutation of cluster indices indicating the processing order for this chain.
        
        x : pd.DataFrame
            Feature matrix of shape (n_samples, n_features).
        
        y : pd.DataFrame
            Multi-label target matrix of shape (n_samples, n_labels).

        Returns
        -------
        None

        Example
        -------
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> lccml = LCCML(model=model, n_chains=1)
        >>> clusters = [['label_a', 'label_b'], ['label_c'], ['label_d', 'label_e']]
        >>> lccml.fit(X_train, Y_train, clusters)
        >>> 
        >>> # Suppose we manually want to train one chain:
        >>> order = lccml.orders[0]  # e.g., array([2, 0, 1])
        >>> lccml.__fitChain(order, X_train, Y_train)

        After execution:
        >>> lccml.chains[-1]  # last trained chain (list of trained models)
        >>> lccml.chain_train_times[-1]  # training time of this chain
        >>> lccml.chain_model_sizes[-1]  # model size (bytes) of this chain
        """
        start_time = time.time()

        chain_X = x.copy()
        chain = []
        self.orderLabelsDataset = y.columns

        for i in order:
            cluster_labels = self.clusters[i]
            chainModel = self.__getModel()

            chain_Y = pd.DataFrame(y[y.columns[cluster_labels]])
            chainModel.labelName_ = y.columns[cluster_labels]

            if chain_Y.shape[1] == 1:
                chainModel.fit(chain_X, chain_Y.values.ravel())
            else:
                chainModel.fit(chain_X, chain_Y)

            chain_X = pd.concat([chain_X, chain_Y], axis=1)
            chain.append(chainModel)

        end_time = time.time()
        self.chain_train_times.append(end_time - start_time)
        self.chains.append(chain)

        model_bytes = pickle.dumps(chain)
        model_size = len(model_bytes)
        self.chain_model_sizes.append(model_size)
        

    # ----------------------------------------------- #
    #                                                 #
    # ----------------------------------------------- #
    def __predictChain(self, x, chainIndex):
        """
        Predict using a single label cluster chain of the ensemble.

        For the specified chain, the model sequentially predicts the probabilistic outputs
        of each cluster, using the predictions of previous clusters as additional features 
        for subsequent clusters.

        Parameters
        ----------
        x : pd.DataFrame
            Feature matrix of shape (n_samples, n_features).
        
        chainIndex : int
            Index of the chain to use for prediction (from 0 to n_chains - 1).

        Returns
        -------
        pd.DataFrame
            Predicted probabilities for all labels from this chain.
            The columns are aligned to the original label set used during training.

        Example
        -------
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> lccml = LCCML(model=model, n_chains=3)
        >>> clusters = [['label_a', 'label_b'], ['label_c'], ['label_d']]
        >>> lccml.fit(X_train, Y_train, clusters)

        >>> # Predict using only chain 0
        >>> pred_chain_0 = lccml.__predictChain(X_test, chainIndex=0)
        >>> print(pred_chain_0.shape)  # (n_samples, n_labels)

        Notes
        -----
        - The method outputs probabilities, not binary predictions.
        - Used internally by `predict()`, which averages predictions across chains.
        """
        chain_X = x.copy()
        predictions = pd.DataFrame([])

        for model in self.chains[chainIndex]:
            pred_proba = model.predict_proba(chain_X)

            if isinstance(pred_proba, list):
                pred_proba = np.concatenate(
                    [probs[:, 1].reshape(-1, 1) for probs in pred_proba],
                    axis=1
                )
            else:
                pred_proba = pred_proba[:, 1].reshape(-1, 1)

            pred_df = pd.DataFrame(pred_proba, columns=model.labelName_)
            predictions[model.labelName_] = pred_df

            # Atualiza as features com as predições do cluster atual
            chain_X = pd.concat([chain_X, pred_df], axis=1)

        # Ordena as colunas de acordo com as labels originais
        predictions = predictions[self.orderLabelsDataset]
        return predictions    
          


    # ----------------------------------------------- #
    #                                                 #
    # ----------------------------------------------- # 
    def __getModel(self):
        """
        Clone the base classifier.

        This method creates a new independent instance of the base model provided
        during the initialization of the LCCML object. It ensures that each model 
        in the chain is trained independently and avoids parameter sharing 
        between chains or clusters.

        Returns
        -------
        sklearn-like estimator
            A fresh copy of the base classifier, ready to be trained.

        Example
        -------
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier(n_estimators=100)
        >>> lccml = LCCML(model=model, n_chains=5)
        
        >>> cloned_model = lccml.__getModel()
        >>> cloned_model.fit(X_train, Y_train)
        >>> predictions = cloned_model.predict(X_test)
        
        Notes
        -----
        - Internally uses `sklearn.base.clone` to copy hyperparameters and structure.
        - This method ensures no data leakage or unintended parameter sharing across models.
        """
        return clone(self.model)
    

    # ----------------------------------------------- #
    #                                                 #
    # ----------------------------------------------- #
    def __preprocessClustersName(self, clusters, Y):
        """
        Convert cluster label names to column indices.

        This function maps each label name inside the provided clusters into the corresponding 
        column indices of the label matrix Y. Internally, all clusters are stored as index 
        positions for efficient access during training and prediction.

        Parameters
        ----------
        clusters : list of lists of str
            Each sublist contains label names that belong to a cluster. 
            Example: [['label_a', 'label_b'], ['label_c'], ['label_d', 'label_e']]
        
        Y : pd.DataFrame
            The multi-label target matrix where columns are label names.

        Returns
        -------
        list of lists of int
            The clusters with label indices instead of names. Each sublist contains 
            the integer indices of the labels in the Y columns.

        Example
        -------
        >>> Y = pd.DataFrame(columns=['label_a', 'label_b', 'label_c', 'label_d'])
        >>> clusters = [['label_a', 'label_b'], ['label_c'], ['label_d']]
        >>> lccml = LCCML(model=SomeClassifier())
        >>> cluster_indices = lccml.__preprocessClustersName(clusters, Y)
        >>> print(cluster_indices)
        [[0, 1], [2], [3]]

        Notes
        -----
        - The column indices are necessary for efficient internal processing.
        - This function assumes that all provided label names exist as columns in Y.
        - Raises `KeyError` if any label name is not found in Y.
        """
        clustersIndexes = [
            [Y.columns.get_loc(label) for label in cluster_labels]
            for cluster_labels in clusters
        ]
        return clustersIndexes
    

    # ----------------------------------------------- #
    #                                                 #
    # ----------------------------------------------- #
    def predict(self, x):
        """
        Performs model prediction by aggregating the results from multiple chains.

        For each chain, the model sequentially predicts the clusters according to its specific order.
        The individual predictions from all chains are aggregated by averaging the predictions 
        across chains for each label. This produces probabilistic predictions for each label.

        Parameters
        ----------
        x : pd.DataFrame
            DataFrame containing the input data (features) for prediction.
            Shape: (n_samples, n_features)

        Returns
        -------
        pd.DataFrame
            DataFrame containing the aggregated predicted probabilities for each label.
            Shape: (n_samples, n_labels)
            The columns match the label names used during training.

        Example
        -------
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> lccml = LCCML(model=model, n_chains=5)
        >>> clusters = [['label_a', 'label_b'], ['label_c'], ['label_d', 'label_e']]
        >>> lccml.fit(X_train, Y_train, clusters)
        >>> Y_pred = lccml.predict(X_test)
        >>> print(Y_pred.head())
        amazed.suprised  happy.pleased  relaxing.calm  quiet.still  ... angry.aggresive

        Notes
        -----
        - The aggregation is performed by computing the mean across predictions from all chains.
        - In case of any error during prediction, an exception will be printed.
        """
        # Performs predictions for each of the chains, using __predictChain for each chain index (0 to n_chains-1)
        # The result of each chain is concatenated along axis 0 (stacked vertically).
        # predictions = pd.concat([self.__predictChain(x, i) for i in range(self.n_chains)], axis=0)        
        
        try:
            predictions = pd.concat([self.__predictChain(x, i) for i in range(self.n_chains)], axis=0)
            # print("PREDICTIONS INSIDE PREDICT 2 before applying the mean")
            # print(predictions)
        except Exception as e:
            print(f"Error: {e}")
        
        predictions_aggregated = predictions.groupby(predictions.index).mean()
        # print("AGGREGATED PREDICTIONS:")
        # print(predictions_aggregated)

        # Now, for each class (column) in the DataFrame, compute the mean of the 
        # predictions for each group of examples (rows)
        # We group by rows (index), and for each group (class), apply the mean        
        # return predictions.groupby(predictions.index).apply(np.mean)
        # predictions.mean(axis=0)  # Computes the mean per label (column)

        # Now, for each sample, we select the highest prediction across chains
        # We group by samples (index), and for each group (class), apply the max
        # predictions_aggregated = predictions.groupby(predictions.index).max()

        return predictions_aggregated
