from dash import dcc
from dash import html
from sklearn.cluster import KMeans
from sklearn import cluster, metrics
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report

import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

class Clustering():

    # Constructeur de la classe clustering
    def __init__(self, df, varX, varY, n_clusters):
        self.df = df
        self.varX = varX
        self.varY = varY
        self.n_clusters = n_clusters
        self.dfX = self.df[self.varX]
        self.dfY = self.df[self.varY]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.dfX, self.dfY, test_size=0.2, random_state=5)

    # Algorithme des KMeans
    def algo_kmeans(self, df, varX, varY, n_clusters):

        km = KMeans(n_clusters=n_clusters)
        y_predicted = km.fit_predict(self.x_train)
        print(y_predicted)
        print(pd.crosstab(y_predicted, self.y_test))

        cross_validation = cross_val_score(KMeans(n_clusters=self.n_clusters), self.x_train, self.y_train, cv=3, scoring='adjusted_rand_score')


        kmeans_algo_layout = html.Br(), html.Div(children=[html.H5("Présentation de l'algorithme des kmeans", style={'textAlign': 'center'})])

        return kmeans_algo_layout

        

        # kmeans = cluster.KMeans(n_clusters=self.n_clusters)
        # kmeans.fit(x_train)

        # print(np.unique(kmeans.labels_, return_counts=True))
        # print(pd.crosstab(kmeans.labels_, dfY))
        
        # print(class_distribution(dfY, relabel))

    

    
    