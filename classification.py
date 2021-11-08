from dash import dcc
from dash import html
from sklearn.cluster import KMeans
from sklearn import cluster, metrics
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report

import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px

class Classification():

    #############################################################
    #            CONSTRUCTEUR CLASSE CLASSIFICATION             #
    #############################################################

    def __init__(self, df, varX, varY, n_clusters):
        
        self.df = df
        self.varX = varX
        self.varY = varY
        self.n_clusters = n_clusters
        self.dfX = df[varX]
        self.dfY = df[varY]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.dfX, self.dfY, test_size=0.2, random_state=5)

    #############################################################
    #              MACHINE A VECTEURS DE SUPPORT                #
    #############################################################

    def algo_svm(self, df, varX, varY, n_clusters):

        # ENTRAINEMENT #

        # PREDICTION #

        # AFFICHAGE #

        svm_algo_layout = html.Div(children=
            [
                html.Hr(),
                html.Br(), 
                html.Div(children=
                    [
                        html.H5("Présentation de l'algorithme de l'arbres de décisions", style={'textAlign': 'center'}),
                        html.Br(),

                        html.P("L'algorithme des K-means vous permet de visualiser comment votre dataframe créer des groupes d'individus. Vous pourrez grâce à la sélection de clusters relancer l'algorithme qui définira au mieux vos différents groupes."),

                        #dcc.Graph(figure=scatter_plot, style={'width':'50%'}),

                        html.P("Afin de vous assister dans le choix de vos clusters, nous vous affichons l'inertie obtenue après chaque itération sur la méthode des K-means."),
                        html.P("La règle est simple : trouver le 'coude' qui permet de définir le nombre de clusters que votre algorithme devra comporter."),
                        #dcc.Graph(figure=plot_line, style={'width': '70%'})
                    ]
                ),

                
            ]

        )

        return svm_algo_layout

    #############################################################
    #            ANALYSE DISCRIMINANTE LINEAIRE                 #
    #############################################################



    #############################################################
    #               REGRESSION LOGISTIQUE                       #
    #############################################################
    