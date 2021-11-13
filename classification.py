from dash import html
from dash import dash_table, dcc
from sklearn import cluster, metrics
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px

import os
import pandas as pd
import time
import plotly.express as px


from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, RepeatedKFold, train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

class Classification():

    #############################################################
    #            CONSTRUCTEUR CLASSE CLASSIFICATION             #
    #############################################################

    def __init__(self, df, varX, varY, t_test):
        self.df = df
        self.varX = varX
        dict_X = []
        for i in range(0, len(self.varX)):
            dict_X.append(list(varX[i].values()))
        liste_X = []
        for i in range(0, len(dict_X)):
            liste_X.append(dict_X[i][0])
        self.varY = varY
        self.dfX = df[liste_X]
        self.dfX_quanti = self.dfX.select_dtypes(include=[np.number])
        self.dfX_quali = self.dfX.select_dtypes(exclude=[np.number])
        self.dfY = df[varY]
        self.t_test = t_test

    #############################################################
    #                     ARBRE DE DECISION                     #
    #############################################################
    
    def algo_arbre(self, nb_feuilles, nb_individus, nb_splits, nb_repeats, standardisation):
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
        arbre = DecisionTreeClassifier(max_depth = nb_feuilles, max_leaf_nodes = nb_individus, random_state = 5)
       
        # ------------ C) Validation croisée -----------
        Valid_croisee = RepeatedKFold(n_splits = nb_splits, n_repeats = nb_repeats, random_state = 0)
        
        start = time.time()
        scores = cross_val_score(arbre, X_ok, self.dfY, cv = Valid_croisee)
        end = time.time()
       
        temps = round((end - start), 2)

        # PLOT DE LA VALIDATION CROISEE #
        K_range = []
        for i in range(1, (nb_repeats*nb_splits)+1):
            K_range.append(i)
        scores_table = pd.DataFrame(data={'K_range': K_range, 'scores': scores})
        plot_line = px.line(scores_table, x="K_range", y="scores")
        
        # ------------ D) Estimation ponctuelle -----------
        XTrain,XTest,yTrain,yTest = train_test_split(X_ok, self.dfY, test_size = self.t_test, random_state = 42)
       
        model = arbre.fit(XTrain, yTrain)
        ypred = model.predict(XTest)

        # IMPORTANCE DE VARIABLES #
        importance_var = {"Variables":X_ok.columns, "Importance des variables":arbre.feature_importances_}
        importance_var = pd.DataFrame(importance_var).sort_values(by="Importance des variables", ascending=False)

        # METRIQUES #
        mc = pd.crosstab(yTest,ypred)
        new_mc = confusion_matrix(yTest, ypred)
        tx_reconaissance = (sum(np.diag(new_mc)) / len(XTest)) * 100  

        # AFFICHAGE DU LAYOUT #
        arbre_algo_layout = html.Div(children=
            [
                html.Hr(),
                html.Br(), 
                html.Div(children=
                    [
                        html.H4("Présentation de l'algorithme de l'arbre de décision", style={'textAlign': 'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        html.Br(),

                        html.P("L'algorithme de l'arbre de décision vous permet de visualiser comment votre dataframe classe vos différents individus. Vous pourrez grâce aux différents paramètres relancer l'algorithme qui classifiera au mieux vos individus."),

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
                            columns=[{'name': i, 'id': i} for i in arbre.classes_],
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
                        html.H5("Graphe de l'évolution du taux de reconnaissance en validation croisée", style={'textAlign':'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        dcc.Graph(figure=plot_line, style={'width': '70%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                        html.Br(),
                        html.Div(children=
                            [

                                html.Span("Taux de reconaissance en % sur le dataset en estimation ponctuelle : ", style={'fontWeight':'bold'}),
                                html.Div(round(tx_reconaissance, 2)),
                                html.Br(),
                                html.Span("Taux de reconaissance en % en validation croisée : ", style={'fontWeight':'bold'}),
                                html.Div(round((scores.mean()*100.0), 2)),
                                html.Br(),
                                html.Span("Temps d'exécution de l'algorithme en validation croisée en secondes : ", style={'fontWeight':'bold'}),
                                html.Div(temps)
                            ]
                        )

                    ]
                ),   
            ]
        )

        return arbre_algo_layout

    #############################################################
    #            ANALYSE DISCRIMINANTE LINEAIRE                 #
    #############################################################
    
    def algo_ADL(self, solv, nb_splits, nb_repeats, standardisation):
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
        lda = LinearDiscriminantAnalysis(solver=solv, n_components = 2)
        
        # ------------ C) Validation croisée -----------
        Valid_croisee = RepeatedKFold(n_splits = nb_splits, n_repeats = nb_repeats, random_state = 0)

        start = time.time()
        scores = cross_val_score(lda, X_ok, self.dfY, cv = Valid_croisee)
        end = time.time()

        temps = round((end - start), 2)

        # PLOT DE LA VALIDATION CROISEE #
        K_range = []
        for i in range(1, (nb_repeats*nb_splits)+1):
            K_range.append(i)
        scores_table = pd.DataFrame(data={'K_range': K_range, 'scores': scores})
        plot_line = px.line(scores_table, x="K_range", y="scores")
        
        # ------------ D) Estimation ponctuelle -----------
        XTrain,XTest,yTrain,yTest = train_test_split(X_ok, self.dfY, test_size = self.t_test, random_state = 42)
        
        model = lda.fit(XTrain, yTrain)
        predLda = model.predict(XTest)
        
        # Structure temporaire pour affichage des coefficients
        tmp = pd.DataFrame(lda.coef_.transpose(), columns = lda.classes_, index = X_ok.columns)
        
        Aff = lda.fit(XTest, yTest).transform(XTest)
        Aff_df = pd.DataFrame(Aff, columns=["Axe1","Axe2"]) 
        Aff_df["yPred"] = predLda
        
        # Affichage
        plot_scatter = px.scatter(Aff_df, x="Axe1", y="Axe2", color='yPred')

        # METRIQUES #
        mc = pd.crosstab(yTest,predLda)
        new_mc = confusion_matrix(yTest, predLda)
        tx_reconaissance = (sum(np.diag(new_mc)) / len(XTest)) * 100

        # AFFICHAGE DU LAYOUT #
        adl_algo_layout = html.Div(children=
            [
                html.Hr(),
                html.Br(), 
                html.Div(children=
                    [
                        html.H4("Présentation de l'algorithme de l'analyse discriminante linéaire", style={'textAlign': 'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        html.Br(),

                        html.P("L'algorithme de l'analyse discriminante linéaire vous permet de visualiser comment votre dataframe classe vos différents individus. Vous pourrez grâce à la sélection de feuilles et du nombre maximum d'individus par feuille relancer l'algorithme qui classifiera au mieux vos individus."),

                        # html.Div(dash_table.DataTable(
                        #     id='importance',
                        #     data=importance_var.to_dict('records'),
                        #     columns=[{'name': i, 'id': i} for i in importance_var.columns],
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
                        # ),style={'margin-right':'200px', 'margin-left':'200px'}),
                        html.Br(),
                        html.P("Enfin, nous vous affichons la matrice de confusion avec la métrique suivante  : le taux d'erreur."),
                        html.H5("Matrice de confusion", style={'textAlign':'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        html.Div(dash_table.DataTable(
                            id='matrice_de_confusion',
                            data=mc.to_dict('records'),
                            columns=[{'name': i, 'id': i} for i in lda.classes_],
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
                        html.H5("Graphe de l'évolution du taux de reconnaissance en validation croisée", style={'textAlign':'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        dcc.Graph(figure=plot_line, style={'width': '70%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                        html.Br(),
                        html.Div(children=
                            [
                                html.Span("Taux de reconaissance en % en validation croisée : ", style={'fontWeight':'bold'}),
                                html.Div(round((scores.mean()*100.0), 2)),
                                html.Br(),
                                html.Span("Temps d'exécution de l'algorithme en validation croisée en secondes : ", style={'fontWeight':'bold'}),
                                html.Div(temps)
                            ]
                        ),
                        html.H5("Visualisation ponctuelle", style={'textAlign':'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        dcc.Graph(figure=plot_scatter, style={'width': '70%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                        html.Br(),
                        html.Div(children=
                            [
                                html.Span("Taux de reconaissance en % sur le dataset en estimation ponctuelle : ", style={'fontWeight':'bold'}),
                                html.Div(round(tx_reconaissance, 2)),
                                html.Br()
                            ]
                        )

                    ]
                ),   
            ]
        )

        return adl_algo_layout

    #############################################################
    #               REGRESSION LOGISTIQUE                       #
    #############################################################
    