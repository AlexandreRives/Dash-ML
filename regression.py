from dash import dash_table, dcc
from dash import html
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

class Regression():

    #Constructeur de la classe Regression
    def __init__(self, df, varX, varY):
        self.df = df
        self.varX = varX
        self.varY = varY
        self.dfX = self.df[self.varX]
        self.dfY = self.df[self.varY]
        if(len(self.df) <= 200):
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.dfX, self.dfY, test_size=0.2, random_state=5)
            self.y_train_disj = pd.get_dummies(self.y_train)
            self.y_test_disj = pd.get_dummies(self.y_test)
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.dfX, self.dfY, test_size=0.33, random_state=5)

    def regression_lineaire_multiple(self, df, varX, varY):

        #############################################################
        #                           ENTRAINEMENT                    #
        #############################################################

        #Instanciation de l'objet de la régression linéaire multiple
        reg_lin_mul = LinearRegression()
        reg_lin_mul.fit(self.x_train, self.y_train_disj)

        #Coefficients
        coeff_reg_lin_mul = reg_lin_mul.coef_
        
        #Colonnes
        colonnes = self.varX

        #Dataframe Coeff + colonnes
        dataframeCC = pd.DataFrame(coeff_reg_lin_mul, columns=colonnes)

        #############################################################
        #                           PREDICTION                      #
        #############################################################

        # Prédiciton
        y_pred = reg_lin_mul.predict(self.x_test)
        print(y_pred)

        # Moindres Carrés Ordinaires (MCO)
        mco = mean_squared_error(self.y_test_disj, y_pred)
        print(mco)

        # Score R2
        r2 = r2_score(self.y_test_disj, y_pred)
        print(r2)

        # Layout
        reg_mult_layout = html.Div(children=[
            html.Br(),
            html.Div(children=[html.H5("Présentation de l'algorithme de la regression linéaire multiple", style={'textAlign': 'center', 'margin-top': '20px'})]),
            html.Br(),
            html.Div(children=[html.P('Liste des coefficients pour les variables sélectionnées : ', style={'margin-left':'30px', 'fontWeight':'bold'}),
            html.Div(dash_table.DataTable(
                id='coefficients',
                data=dataframeCC.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in dataframeCC.columns],
                style_cell={'textAlign':'center'},
                style_header={'fontWeight':'bold'}
            ), style={'margin':'20px'})
            ])

        ])

        return reg_mult_layout

        

        # kmeans = cluster.KMeans(n_clusters=self.n_clusters)
        # kmeans.fit(x_train)

        # print(np.unique(kmeans.labels_, return_counts=True))
        # print(pd.crosstab(kmeans.labels_, dfY))
        
        # print(class_distribution(dfY, relabel))

    
    