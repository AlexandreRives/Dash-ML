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
from sklearn.neighbors import KNeighborsRegressor


import matplotlib.pyplot as plt
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import time

class Regression():
    #############################################################
    #             CONSTRUCTEUR CLASSE REGRESSION                #
    #############################################################

    def __init__(self, df, varX, varY, t_test):
        self.df = df
        self.varX = varX
        self.varY = varY
        self.dfX = df[varX]
        self.dfX_quanti = self.dfX.select_dtypes(include=[np.number])
        self.dfX_quali = self.dfX.select_dtypes(exclude=[np.number])
        self.dfY = df[varY]
        self.t_test = t_test


    def algo_knn(self, K, nb_splits, nb_repeats, standardisation):
        # ------------ A) Traitement du dataset ------------
        
        # Centrer-réduire les variables quantitatives si standardisation = Oui
        if standardisation == 'Oui' :
            sc = StandardScaler()
            X_quant = pd.DataFrame(sc.fit_transform(self.dfX_quanti))
        else : 
            X_quant = self.dfX_quanti

        # Recoder les variables qualitatives
        if self.dfX_quali.shape[1] == 0:
            X_ok = X_quant
        else:
            X_qual = pd.get_dummies(self.dfX_quali, drop_first=True)
            # Concatener X_qual et X_quant 
            X_ok = pd.concat([X_quant,X_qual], axis = 1)

        # ------------ B) Instanciation -----------
        knn = KNeighborsRegressor(K)

        # ------------ C) Validation croisée -----------
        Valid_croisee = RepeatedKFold(n_splits = nb_splits, n_repeats = nb_repeats, random_state = 0)
        
        start = time.time()
        scores = cross_val_score(knn, X_ok, self.dfY, cv = Valid_croisee)
        end = time.time()

        temps = round((end - start), 2)

        # ------------ D) Estimation ponctuelle -----------
        XTrain,XTest,yTrain,yTest = train_test_split(X_ok, self.dfY, test_size= self.t_test)
        print(XTrain)

        knn.fit(XTrain, yTrain)
        y_pred = knn.predict(XTest)
        
        scores_table = pd.DataFrame(data={'yTest': yTest, 'y_pred': y_pred})

        # ESTIMATEURS #
        mse = mean_squared_error(yTest, y_pred)

        # GRAPHIQUES #
        plot_scatter = px.scatter(scores_table, x="yTest", y="y_pred")
        
        knn_algo_layout = html.Div(children=
            [
                html.Hr(),
                html.Br(), 
                html.Div(children=
                    [
                        html.H4("Présentation de l'algorithme des K plus proches voisins", style={'textAlign': 'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        html.Br(),

                        html.P("L'algorithme des K plus proches voisins vous permet de visualiser comment votre dataframe prédit vos différents individus. Vous pourrez grâce aux différents paramètres relancer l'algorithme qui prédira au mieux vos individus."),

                        # html.H5("Tableau d'importance des variables", style={'textAlign':'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        # html.Br(),

                        # html.P("Nous effectuons pour vous une analyse préliminaire qui vous permet de choisir le nombre de variable à retenir. Ci-dessous le tableau de l'importance des variables : "),

                        # html.Br(),
                        #html.P("Enfin, nous vous affichons la matrice de confusion avec la métrique suivante  : le taux d'erreur."),
                        #html.H5("Matrice de confusion", style={'textAlign':'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        # html.Div(dash_table.DataTable(
                        #     id='matrice_de_confusion',
                        #     data=mc.to_dict('records'),
                        #     columns=[{'name': i, 'id': i} for i in model.classes_],
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
                        html.Br(),
                        html.H5("Graphique de comparaison des valeurs observées et valeurs prédites", style={'textAlign':'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        dcc.Graph(figure=plot_scatter, style={'width': '30%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                        html.Br(),
                        html.Div(children=
                            [
                                html.Span("Mean Squared Error : ", style={'fontWeight':'bold'}),
                                html.Div(round(mse, 2)),
                                html.Br(),
                                # html.Span("Taux de reconnaissance en % en validation croisée : ", style={'fontWeight':'bold'}),
                                # html.Div(round((scores.mean()*100.0), 2)),
                                # html.Br(),
                                html.Span("Temps d'exécution de l'algorithme en validation croisée en secondes : ", style={'fontWeight':'bold'}),
                                html.Div(temps)
                            ]
                        )

                    ]
                ),   
            ]
        )

        return knn_algo_layout


    #############################################################
    #              CONSTRUCTEUR CLASSE REGRESSION               #
    #############################################################

    # def __init__(self, df, varX, varY):

    #     self.df = df
    #     self.varX = varX
    #     self.varY = varY
    #     self.dfX = self.df[self.varX]
    #     self.dfY = self.df[self.varY]
    #     self.x_disj = pd.get_dummies(self.dfX, drop_first=True)
    #     self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_disj, self.dfY, test_size=0.33, random_state=5)

    # #############################################################
    # #               REGRESSION LINEAIRE MULTIPLE                #
    # #############################################################

    # def regression_lineaire_multiple(self, nb_variables, nb_splits, nb_repeats):

    #     if(len(self.x_disj.columns) < nb_variables):

    #         # ENTRAINEMENT #
    #         reg_lin_mul = LinearRegression()
    #         model = reg_lin_mul.fit(self.x_train, self.y_train)
            
    #         # PREDICTION #
    #         y_pred = model.predict(self.x_test)

    #         # ESTIMATEURS #
    #         mse = mean_squared_error(self.y_test, y_pred)

    #         # COEFFICIENTS #
    #         coeff_reg_lin_mul = model.coef_
    #         dataframeCC = pd.Series(coeff_reg_lin_mul)

    #     else:
    #         # SELECTION DE VARIABLES #
    #         selector = SelectKBest(chi2, k = nb_variables)
    #         selector.fit_transform(self.x_train, self.y_train)

    #         # ENTRAINEMENT #
    #         reg_lin_mul = LinearRegression()
    #         x_train_kbest = self.x_train.loc[:,selector.get_support()]
    #         x_test_kbest = self.x_test.loc[:,selector.get_support()]
    #         reg_lin_mul.fit(x_train_kbest, self.y_train)

    #         # PREDICITON #
    #         ypred = reg_lin_mul.predict(x_test_kbest)

    #         # ESTIMATEURS #
    #         mco = mean_squared_error(self.y_test, ypred)

    #         # CONSTRUCTION DU GRAPHE #
    #         scores_table = pd.DataFrame(data={'y_test': self.y_test, 'ypred': ypred})
    #         plot_scatter = px.scatter(scores_table, x="y_test", y="ypred")

    #         # COEFFICIENTS #
    #         coeff_reg_lin_mul = reg_lin_mul.coef_
    #         dataframeCC = pd.Series(coeff_reg_lin_mul)

    #         # VALIDATION CROISEE + CALCUL DU TEMPS #
    #         start = time.time()
    #         Valid_croisee = RepeatedKFold(n_splits = nb_splits, n_repeats = nb_repeats, random_state = 0)
    #         scores = cross_val_score(reg_lin_mul, self.x_disj, self.dfY, cv = Valid_croisee)
    #         end = time.time()
    #         diff_time = round((end - start), 2)
    #         scores_moyen = round(scores.mean()*100, 2)

    #     # AFFICHAGE #
    #     reg_mult_layout = html.Div(children=
    #         [
    #             html.Br(),
    #             html.Hr(),
    #             html.Div(children=
    #                 [
    #                     html.H5("Présentation de l'algorithme de la regression linéaire multiple", style={'textAlign': 'center', 'margin-top': '20px'})
    #                 ]
    #             ),
    #             html.Br(),
    #             html.Div(children=
    #                 [
    #                     html.P('Liste des coefficients pour les variables sélectionnées : ', style={'margin-left':'30px', 'fontWeight':'bold'}),
    #                     # html.Div(dash_table.DataTable(
    #                     #     id='coefficients',
    #                     #     data=dataframeCC.to_dict('records'),
    #                     #     columns=[{'name': i, 'id': i} for i in dataframeCC.columns],
    #                     #     style_header={
    #                     #         'backgroundColor': 'rgb(30, 30, 30)',
    #                     #         'color': 'white',
    #                     #     },
    #                     #     style_cell={'textAlign':'center', 
    #                     #                 'overflow': 'hidden',
    #                     #                 'textOverflow': 'ellipsis',
    #                     #                 'maxWidth': 0},
    #                     #     style_data={'backgroundColor': 'rgb(50, 50, 50)',
    #                     #                 'color': 'white',
    #                     #                 'width': '20px'
    #                     #     },
    #                     # ),style={'margin':'20px'}),
    #                     dcc.Graph(figure=plot_scatter, style={'width': '30%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
    #                 ]
    #             ),
    #             html.Div(children=
    #                         [
    #                             html.Span("Taux de reconnaissance en % en validation croisée : ", style={'fontWeight':'bold'}),
    #                             html.Div(scores_moyen),
    #                             html.Br(),
    #                             html.Span("Temps d'execution de l'algorithme en validation croisée en seconde : ", style={'fontWeight':'bold'}),
    #                             html.Div(diff_time)
    #                         ]
    #             )
    #         ]
    #     )

    #     return reg_mult_layout

    # #############################################################
    # #                   K PLUS PROCHES VOISINS                  #
    # #############################################################








    
    