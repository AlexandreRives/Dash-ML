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
    #            CONSTRUCTEUR CLASSE CLUSTERING                 #
    #############################################################

    def __init__(self, df, varX, varY, n_clusters):
        
        self.varX = varX
        self.varY = varY
        self.n_clusters = n_clusters
        self.dfX = df[varX]
        self.dfY = df[varY]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.dfX, self.dfY, test_size=0.2, random_state=5)

    #############################################################
    #                        K-MEANS                            #
    #############################################################

    def algo_kmeans(self, df, varX, varY, n_clusters):

        # ENTRAINEMENT #
        
        model = KMeans(n_clusters=n_clusters)
        clusters = model.fit(self.dfX)
        

        # Construction du graphe avec la méthode du coude pour assister l'utilisateur dans son choix de clusters
        inertie = []
        K_range = range(1,11)
        inertie_table = []
        for k in K_range:
            model = KMeans(n_clusters=k).fit(self.dfX)
            inertie.append(model.inertia_)
            inertie_table.append([k, model.inertia_])
        
        inertie_table = pd.DataFrame(inertie_table, columns=['K_range', 'inertie'])
        
        plot_line = px.line(inertie_table, x="K_range", y="inertie", title="Choix du nombre de clusters")

        # Construction du graphique
        df_scatter = pd.DataFrame(self.dfX)
        print(clusters.cluster_centers_)

        # Affichage du graphique avec les différents clusters et centroïdes
        #scatter_plot = px.scatter(clusters.cluster_centers_[:,0], model.cluster_centers_[:,1])

        #cross_validation = cross_val_score(KMeans(n_clusters=self.n_clusters), self.x_train, self.y_train, cv=3, scoring='adjusted_rand_score')

        # AFFICHAGE #

        kmeans_algo_layout = html.Div(children=
            [
                html.Hr(),
                html.Br(), 
                html.Div(children=
                    [
                        html.H5("Présentation de l'algorithme des K-means", style={'textAlign': 'center'}),
                        html.Br(),

                        html.P("L'algorithme des K-means vous permet de visualiser comment votre dataframe créer des groupes d'individus. Vous pourrez grâce à la sélection de clusters relancer l'algorithme qui définira au mieux vos différents groupes."),

                        #dcc.Graph(figure=scatter_plot, style={'width':'50%'}),

                        html.P("Afin de vous assister dans le choix de vos clusters, nous vous affichons l'inertie obtenue après chaque itération sur la méthode des K-means."),
                        html.P("La règle est simple : trouver le 'coude' qui permet de définir le nombre de clusters que votre algorithme devra comporter."),
                        dcc.Graph(figure=plot_line, style={'width': '70%'})
                    ]
                ),

                
            ]
        )

        return kmeans_algo_layout

        # kmeans = cluster.KMeans(n_clusters=self.n_clusters)
        # kmeans.fit(x_train)

        # print(np.unique(kmeans.labels_, return_counts=True))
        # print(pd.crosstab(kmeans.labels_, dfY))
        
        # print(class_distribution(dfY, relabel))

    

    
    