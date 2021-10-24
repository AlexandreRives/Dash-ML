from dash import dcc
from dash import html
import dash_bootstrap_components as dbc

from sklearn import cluster, metrics
from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

class KMeans_algo():

    def __init__(self, df, varX, varY, n_clusters):
        self.df = df
        self.varX = varX
        self.varY = varY
        self.n_clusters = n_clusters

    def Algo_KMeans(self, df, varX, varY, n_clusters):

        print(n_clusters)
        print(df)
        print(varX)
        print(varY)


        # Faire un nuage par pair pour voir s'il y a un lien entre certaines variables.


        # kmeans = cluster.KMeans(n_clusters=n_clusters)
        # kmeans.fit(dfX)

        # #Vérification de l'affectation des individus par rapport aux classes
        # np.unique(kmeans.labels_, return_counts=True)

    
    