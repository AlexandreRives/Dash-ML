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

    def regression_lineaire_multiple(self, df, varX, varY, nb_variables, nb_splits, nb_repeats):

        if(len(self.x_disj.columns) < nb_variables):

            # ENTRAINEMENT #
            reg_lin_mul = LinearRegression()
            model = reg_lin_mul.fit(self.x_train, self.y_train)
            
            # PREDICTION #
            y_pred = model.predict(self.x_test)

            # ESTIMATEURS #
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)

            # COEFFICIENTS #
            coeff_reg_lin_mul = model.coef_
            dataframeCC = pd.Series(coeff_reg_lin_mul)

        else:
            # SELECTION DE VARIABLES #
            selector = SelectKBest(chi2, k = nb_variables)
            selector.fit_transform(self.x_train, self.y_train)

            # ENTRAINEMENT #
            reg_lin_mul = LinearRegression()
            x_train_kbest = self.x_train.loc[:,selector.get_support()]
            x_test_kbest = self.x_test.loc[:,selector.get_support()]
            reg_lin_mul.fit(x_train_kbest, self.y_train)

            # PREDICITON #
            ypred = reg_lin_mul.predict(x_test_kbest)

            # ESTIMATEURS #
            mco = mean_squared_error(self.y_test, ypred)
            r2 = r2_score(self.y_test, ypred)

            # CONSTRUCTION DU GRAPHE #
            scores_table = pd.DataFrame(data={'y_test': self.y_test, 'ypred': ypred})
            plot_scatter = px.scatter(scores_table, x="y_test", y="ypred")

            # COEFFICIENTS #
            coeff_reg_lin_mul = reg_lin_mul.coef_
            dataframeCC = pd.Series(coeff_reg_lin_mul)

            # VALIDATION CROISEE + CALCUL DU TEMPS #
            start = time.time()
            Valid_croisee = RepeatedKFold(n_splits = nb_splits, n_repeats = nb_repeats, random_state = 0)
            scores = cross_val_score(reg_lin_mul, self.x_train, self.y_train, cv = Valid_croisee)
            end = time.time()
            diff_time = round((end - start), 2)
            scores_moyen = round(scores.mean()*100, 2)

        # AFFICHAGE #
        reg_mult_layout = html.Div(children=
            [
                html.Br(),
                html.Hr(),
                html.Div(children=
                    [
                        html.H5("Présentation de l'algorithme de la regression linéaire multiple", style={'textAlign': 'center', 'margin-top': '20px'})
                    ]
                ),
                html.Br(),
                html.Div(children=
                    [
                        html.P('Liste des coefficients pour les variables sélectionnées : ', style={'margin-left':'30px', 'fontWeight':'bold'}),
                        # html.Div(dash_table.DataTable(
                        #     id='coefficients',
                        #     data=dataframeCC.to_dict('records'),
                        #     columns=[{'name': i, 'id': i} for i in dataframeCC.columns],
                        #     style_header={
                        #         'backgroundColor': 'rgb(30, 30, 30)',
                        #         'color': 'white',
                        #     },
                        #     style_cell={'textAlign':'center', 
                        #                 'overflow': 'hidden',
                        #                 'textOverflow': 'ellipsis',
                        #                 'maxWidth': 0},
                        #     style_data={'backgroundColor': 'rgb(50, 50, 50)',
                        #                 'color': 'white',
                        #                 'width': '20px'
                        #     },
                        # ),style={'margin':'20px'}),
                        dcc.Graph(figure=plot_scatter, style={'width': '30%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                    ]
                ),
                html.Div(children=
                            [
                                html.Span("Taux de précision en % en validation croisée : ", style={'fontWeight':'bold'}),
                                html.Div(scores_moyen),
                                html.Br(),
                                html.Span("Temps d'execution de l'algorithme en validation croisée en seconde : ", style={'fontWeight':'bold'}),
                                html.Div(diff_time)
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
        model = model.fit(self.x_train, self.y_train)

        # IMPORTANCE DE VARIABLES #
        importance_var = {"Variables":varX, "Importance des variables":model.feature_importances_}
        importance_var = pd.DataFrame(importance_var).sort_values(by="Importance des variables", ascending=False)

        # PREDICTION #
        pred = model.predict(self.x_test)

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
        scores_moyen = round(scores.mean()*100, 2)

        # PLOT DE LA VALIDATION CROISEE
        K_range = []
        for i in range(1, (nb_repeats*nb_splits)+1):
            K_range.append(i)
        scores_table = pd.DataFrame(data={'K_range': K_range, 'scores': scores})
        plot_line = px.line(scores_table, x="K_range", y="scores")

        # AFFICHAGE DU LAYOUT
        arbre_algo_layout = html.Div(children=
            [
                html.Hr(),
                html.Br(), 
                html.Div(children=
                    [
                        html.H4("Présentation de l'algorithme de l'arbre de décision", style={'textAlign': 'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        html.Br(),

                        html.P("L'algorithme de l'arbre de décision vous permet de visualiser comment votre dataframe classe vos différents individus. Vous pourrez grâce à la sélection de feuilles et du nombre maximum d'individus par feuille relancer l'algorithme qui classifiera au mieux vos individus."),

                        html.H5("Tableau d'importance des variables", style={'textAlign':'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        html.Br(),

                        html.P("Nous effectuons pour vous une analyse préliminaire qui vous permet de choisir le nombre de variable à retenir. Ci-dessous le tableau de l'importance des variables : "),

                        html.Div(dash_table.DataTable(
                            id='importance',
                            data=importance_var.to_dict('records'),
                            columns=[{'name': i, 'id': i} for i in importance_var.columns],
                            style_header={
                                'backgroundColor': 'rgb(30, 30, 30)',
                                'color': 'white',
                            },
                            style_cell={'textAlign':'center', 
                                        'overflow': 'hidden',
                                        'textOverflow': 'ellipsis',
                                        'maxWidth': 0},
                            style_data={'backgroundColor': 'rgb(50, 50, 50)',
                                        'color': 'white',
                                        'width': '20px'
                            },
                        ),style={'margin-right':'200px', 'margin-left':'200px'}),
                        html.Br(),
                        html.P("Enfin, nous vous affichons la matrice de confusion avec la métrique suivante  : le taux d'erreur."),
                        html.H5("Matrice de confusion", style={'textAlign':'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        html.Div(dash_table.DataTable(
                            id='matrice_de_confusion',
                            data=mc.to_dict('records'),
                            columns=[{'name': i, 'id': i} for i in colonne_mat],
                            style_header={
                                'backgroundColor': 'rgb(30, 30, 30)',
                                'color': 'white',
                            },
                            style_cell={'textAlign':'center', 
                                        'overflow': 'hidden',
                                        'textOverflow': 'ellipsis',
                                        'maxWidth': 0},
                            style_data={'backgroundColor': 'rgb(50, 50, 50)',
                                        'color': 'white',
                                        'width': '20px'
                            },
                        ),style={'margin':'20px'}),
                        html.Br(),
                        html.H5("Graphe de l'évolution du taux de précision en validation croisée", style={'textAlign':'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        dcc.Graph(figure=plot_line, style={'width': '70%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                        html.Br(),
                        html.Div(children=
                            [

                                html.Span("Taux de précision en % sur le dataset en 70/30 : ", style={'fontWeight':'bold'}),
                                html.Div(tx_precision),
                                html.Br(),
                                html.Span("Taux de précision en % en validation croisée : ", style={'fontWeight':'bold'}),
                                html.Div(scores_moyen),
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






    
    