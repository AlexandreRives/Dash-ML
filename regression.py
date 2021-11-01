from dash import dash_table, dcc
from dash import html
from numpy.lib.function_base import select
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import time

class Regression():

    #############################################################
    #              CONSTRUCTEUR CLASSE REGRESSION               #
    #############################################################

    def __init__(self, df, varX, varY):
        self.df = df
        self.varX = varX
        self.varY = varY
        self.dfX = self.df[self.varX]
        self.dfY = self.df[self.varY]
        self.x_disj = pd.get_dummies(self.dfX, drop_first=True)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_disj, self.dfY, test_size=0.33, random_state=5)



    #############################################################
    #               REGRESSION LINEAIRE MULTIPLE                #
    #############################################################

    def regression_lineaire_multiple(self, df, varX, varY, nb_variables):

        if(len(self.x_disj.columns) < nb_variables):
        #On lance l'algo sans sélection de variables

        ##### A COMMENTER ! ######

        # ENTRAINEMENT #
        #Instanciation de l'objet de la régression linéaire multiple
            print("on est pas dans la sélection de variables")
            reg_lin_mul = LinearRegression()
            model = reg_lin_mul.fit(self.x_train, self.y_train)
            #coeff
            coeff_reg_lin_mul = model.coef_
            print(coeff_reg_lin_mul)
            y_pred = model.predict(self.x_test)
            print(y_pred)
            mco = mean_squared_error(self.y_test, y_pred)
            print(mco)
            r2 = r2_score(self.y_test, y_pred)
            print(r2)
        else:
            print("on est dans la sélection de variables")
            #On lance l'algo avec la sélection de variables
            selector = SelectKBest(chi2, k = nb_variables)
            selector.fit_transform(self.x_train, self.y_train)
            reg_lin_mul = LinearRegression()
            x_train_kbest = self.x_train.loc[:,selector.get_support()]
            x_test_kbest = self.x_test.loc[:,selector.get_support()]
            reg_lin_mul.fit(x_train_kbest, self.y_train)
            ypred = reg_lin_mul.predict(x_test_kbest)
            coeff_reg_lin_mul = reg_lin_mul.coef_
            print(ypred)
            mco = mean_squared_error(self.y_test, ypred)
            print(mco)
            r2 = r2_score(self.y_test, ypred)
            print(r2)


        #Dataframe Coeff + colonnes
        dataframeCC = pd.Series(coeff_reg_lin_mul)

        # PREDICTION #


        # ESTIMATEURS #

        # Score R2
        r2 = r2_score(self.y_test, y_pred)
        print(r2)

        # AFFICHAGE #

        reg_mult_layout = html.Div(children=
            [
                html.Br(),
                html.Div(children=
                    [
                        html.H5("Présentation de l'algorithme de la regression linéaire multiple", style={'textAlign': 'center', 'margin-top': '20px'})
                    ]
                ),
                html.Br(),
                html.Div(children=
                    [
                        html.P('Liste des coefficients pour les variables sélectionnées : ', style={'margin-left':'30px', 'fontWeight':'bold'}),
                        html.Div(dash_table.DataTable(
                            id='coefficients',
                            data=dataframeCC.to_dict('records'),
                            columns=[{'name': i, 'id': i} for i in dataframeCC.columns],
                            style_cell={'textAlign':'center'},
                            style_header={'fontWeight':'bold'}
                        ),style={'margin':'20px'})
                    ]
                )
            ]
        )

        return reg_mult_layout

    #############################################################
    #                     ARBRE DE DECISION                     #
    #############################################################

    def algo_arbre(self, df, varX, varY, nb_feuilles, nb_individus, nb_splits, nb_repeats):

        # ENTRAINEMENT #
        model = DecisionTreeClassifier(max_depth=nb_feuilles, max_leaf_nodes=nb_individus, random_state=5)
        model = model.fit(self.x_train_disj, self.y_train)

        # AFFICHAGE #
        arbre_plot = plot_tree(model, feature_names= list(varX), filled=True)

        # IMPORTANCE DE VARIABLES #
        importance_var = {"Variables":varX, "Importance des variables":model.feature_importances_}
        importance_var = pd.DataFrame(importance_var).sort_values(by="Importance des variables", ascending=False)

        # PREDICTION #
        pred = model.predict(self.x_test_disj)

        # MATRICE DE CONFUSION #
        mc = confusion_matrix(self.y_test, pred)
        colonne_mat = set(self.dfY)
        mc = pd.DataFrame(mc, columns=colonne_mat, index=colonne_mat)
        mc = mc.reset_index()
        colonne_mat = mc.columns

        # METRIQUES #
        new_mc = confusion_matrix(self.y_test, pred)
        tx_precision = (sum(np.diag(new_mc)) / len(self.x_test)) * 100

        # VALIDATION CROISEE + CALCUL DU TEMPS #
        start = time.time()

        Valid_croisee = RepeatedKFold(n_splits = nb_splits, n_repeats = nb_repeats, random_state = 0)
        scores = cross_val_score(model, self.x_train, self.y_train, cv = Valid_croisee)
        end = time.time()
        diff_time = round((end - start), 2)
        scores = scores.mean()*100

        arbre_algo_layout = html.Div(children=
            [
                html.Hr(),
                html.Br(), 
                html.Div(children=
                    [
                        html.H5("Présentation de l'algorithme de l'arbre de décision", style={'textAlign': 'center', 'text-decoration': 'underline'}),
                        html.Br(),

                        html.P("L'algorithme de l'arbre de décision vous permet de visualiser comment votre dataframe classe vos différents individus. Vous pourrez grâce à la sélection de feuilles et du nombre maximum d'individus par feuille relancer l'algorithme qui classifiera au mieux vos individus."),

                        html.H5("Tableau d'importance des variables", style={'textAlign':'center', 'text-decoration': 'underline'}),
                        html.Br(),

                        html.P("Nous effectuons pour vous une analyse préliminaire qui vous permet de choisir le nombre de variable à retenir. Ci-dessous le tableau de l'importance des variables : "),

                        html.Div(dash_table.DataTable(
                            id='importance',
                            data=importance_var.to_dict('records'),
                            columns=[{'name': i, 'id': i} for i in importance_var.columns],
                            style_cell={'textAlign':'center'},
                            style_header={'fontWeight':'bold'}
                        ),style={'margin':'20px'}),

                        html.P("Enfin, nous vous afficherons la matrice de confusion avec la métrique suivante  : le taux d'erreur."),
                        html.H5("Matrice de confusion", style={'textAlign':'center', 'text-decoration': 'underline'}),
                        html.Div(dash_table.DataTable(
                            id='matrice_de_confusion',
                            data=mc.to_dict('records'),
                            columns=[{'name': i, 'id': i} for i in colonne_mat],
                            style_cell={'textAlign':'center'},
                            style_header={'fontWeight':'bold'}
                        ),style={'margin':'20px'}),
                        html.Br(),
                        html.Div(children=
                            [
                                html.Span("Taux de précision en % sur le dataset en 70/30 : ", style={'fontWeight':'bold'}),
                                html.Div(tx_precision),
                                html.Br(),
                                html.Span("Taux de précision en % en validation croisée : ", style={'fontWeight':'bold'}),
                                html.Div(scores),
                                html.Br(),
                                html.Span("Temps d'execution de l'algorithme en validation croisée en seconde : ", style={'fontWeight':'bold'}),
                                html.Div(diff_time)
                            ]
                        )

                    ]
                ),   
            ]
        )

        return arbre_algo_layout

    #############################################################
    #                   K PLUS PROCHES VOISINS                  #
    #############################################################






    
    