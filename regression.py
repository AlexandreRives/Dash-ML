from dash import dash_table, dcc
from dash import html
from numpy.lib.function_base import select
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet

import pandas as pd
import numpy as np
import plotly.express as px
import time
import plotly.graph_objects as go

class Regression():
    #############################################################
    #             CONSTRUCTEUR CLASSE REGRESSION                #
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
    #                    K PLUS PROCHES VOISINS                 #
    #############################################################
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
        scores = cross_val_score(knn, X_ok, self.dfY, cv = Valid_croisee, n_jobs=-2)
        end = time.time()

        temps = round((end - start), 2)

        # ------------ D) Estimation ponctuelle -----------
        XTrain,XTest,yTrain,yTest = train_test_split(X_ok, self.dfY, test_size= self.t_test, random_state=42)
        print(XTrain)

        knn.fit(XTrain, yTrain)
        y_pred = knn.predict(XTest)
        
        scores_table = pd.DataFrame(data={'yTest': yTest, 'y_pred': y_pred})

        # ESTIMATEURS #
        rmse = np.sqrt(mean_squared_error(yTest, y_pred))

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
                                html.Span("Racine de l'erreur quadratique moyenne : ", style={'fontWeight':'bold'}),
                                html.Div(round(rmse, 2)),
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

    # #############################################################
    # #                     ELASTIC NET                           #
    # #############################################################   

    def elasticnet(self, l1_ratio, nb_splits, nb_repeats, iterations, alpha, standardisation):
        #### PREPARATION DES DONNEES ####
        
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
        

        # ------------ B) Instanciation du modèle -----------
        
        elc = ElasticNet(alpha = alpha, l1_ratio = l1_ratio, normalize = False, max_iter = iterations)
        
        # ------------ C) Split en échantillons d'apprentissage et de test -----------
        
        XTrain, XTest, yTrain, yTest = train_test_split(X_ok, self.dfY, test_size = self.t_test, random_state = 42)
        #print(Xtrain)

        #Entraînement
        elc.fit(XTrain, yTrain)

        #Prédiction
        y_pred = elc.predict(XTest)

        # ------------ D) Performance -----------

        #Comparaison observations vs prédictions
        scores_table = pd.DataFrame(data={'yTest': yTest, 'y_pred': y_pred})
        
        #Mesure de la racine de l'erreur quadratique moyenne
        RMSE = np.sqrt(mean_squared_error(yTest,y_pred))

        #Mesure du R^2      
        perf = r2_score(yTest, y_pred)
        
        #Coefficients du modèle
        coeff = elc.coef_ 


        # ------------ E) Validation croisée -----------
        cv = RepeatedKFold(n_splits= nb_splits, n_repeats= nb_repeats, random_state=0)
        
        start = time.time()
        scores = cross_val_score(elc, X_ok, self.dfY, cv = cv, n_jobs=-2)
        end = time.time()
        
        temps = round((end - start), 2)

      
        
        # ------------ F) Visualisation -----------
        
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=yTest,
                    mode='markers',
                    name='observations'))
        fig.add_trace(go.Scatter(y=y_pred,
                    mode='lines',
                    name='prédictions'))

        #fig.show()
        
        fig2 = go.Figure(data=go.Scatter(y=scores,
                                        mode='lines+markers',
                                        name='Scores'))
        
        #fig2.show()
        
        elastic_net_layout = html.Div(children=
            [
                html.Hr(),
                html.Br(), 
                html.Div(children=
                    [
                        html.H4("Présentation de l'algorithme de la régression ElasticNet", style={'textAlign': 'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        html.Br(),
                        html.P("L'algorithme de régression ElasticNet vous permet de visualiser comment le modèle prédit vos différents individus. Vous pourrez grâce aux différents paramètres relancer l'algorithme qui prédira au mieux vos individus."),
                        html.Br(),
                        html.P("Le paramètre l1 ratio, compris entre 0 et 1 vous permet d'orienter la pénalité vers Ridge, Lasso ou entre les deux"),
                        html.P("Le paramètre alpha est "),
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
                        dcc.Graph(figure=fig, style={'width': '70%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                        html.Br(),
                        html.H5("Evolution des estimateurs de la validation croisée", style={'textAlign':'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        dcc.Graph(figure = fig2, style={'width': '70%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                        html.Br(),
                        html.Div(children=
                            [
                                html.Span("Racine de l'erreur quadratique moyenne : ", style={'fontWeight':'bold'}),
                                html.Div(round(RMSE, 2)),
                                html.Br(),
                                html.Span("Coefficient de détermination R² : ", style={'fontWeight':'bold'}),
                                html.Div(round(perf, 2)),
                                html.Br(),
                                # html.Span("Coefficients du modèles : ", style={'fontWeight':'bold'}),
                                # html.Div(round(coeff, 2)),
                                html.Span("Temps d'exécution de l'algorithme en validation croisée en secondes : ", style={'fontWeight':'bold'}),
                                html.Div(temps)
                            ]
                        )

                    ]
                ),   
            ]
        )

        return elastic_net_layout


    # #############################################################
    # #               REGRESSION LINEAIRE MULTIPLE                #
    # #############################################################

    def regression_lineaire_multiple(self, nb_splits, nb_repeats, standardisation):

        #### PREPARATION DES DONNEES ####
        
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

        # ------------ B) Instanciation du modèle -----------
        
        RegMul = LinearRegression()

        # ------------ C) Split en échantillons d'apprentissage et de test -----------
        
        XTrain, XTest, yTrain, yTest = train_test_split(X_ok, self.dfY, test_size = self.t_test, random_state = 42)

        #Entraînement
        RegMul.fit(XTrain, yTrain)

        #Prédiction
        y_pred = RegMul.predict(XTest)

        # ------------ D) Performance -----------

        #Comparaison observations vs prédictions
        scores_table = pd.DataFrame(data={'yTest': yTest, 'y_pred': y_pred})
        
        #Mesure de la racine de l'erreur quadratique moyenne
        RMSE = np.sqrt(mean_squared_error(yTest,y_pred))

        #Mesure du R^2      
        perf = r2_score(yTest, y_pred)

        # ------------ E) Validation croisée -----------
        cv = RepeatedKFold(n_splits= nb_splits, n_repeats= nb_repeats, random_state=0)
        
        start = time.time()
        scores = cross_val_score(RegMul, X_ok, self.dfY, cv = cv, n_jobs=-2)
        end = time.time()
        
        temps = round((end - start), 2)

        plot_scatter = px.scatter(scores_table, x="yTest", y="y_pred")

        # AFFICHAGE #
        reg_mult_layout = html.Div(children=
            [
                html.Br(),
                html.Hr(),
                html.Div(children=
                    [
                        html.H5("Présentation de l'algorithme de la regression linéaire multiple", style={'textAlign': 'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'})
                    ]
                ),
                html.Br(),
                html.P("L'algorithme de la régression linéaire multiple vous permet de visualiser comment votre dataframe prédit vos différents individus. Vous pourrez grâce aux différents paramètres relancer l'algorithme qui prédira au mieux vos individus."),
                html.Br(),
                html.Div(children=
                    [
                        #html.P('Liste des coefficients pour les variables sélectionnées : ', style={'margin-left':'30px', 'fontWeight':'bold'}),
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
                        html.H5("Graphique de comparaison des valeurs observées et valeurs prédites", style={'textAlign':'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        dcc.Graph(figure=plot_scatter, style={'width': '30%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                    ]
                ),
                html.Div(children=
                            [
                                html.Span("Racine de l'erreur quadratique moyenne : ", style={'fontWeight':'bold'}),
                                html.Div(round(RMSE, 2)),
                                html.Br(),
                                html.Span("Coefficient de détermination R² : ", style={'fontWeight':'bold'}),
                                html.Div(round(perf, 2)),
                                html.Br(),
                                html.Span("Temps d'execution de l'algorithme en validation croisée en seconde : ", style={'fontWeight':'bold'}),
                                html.Div(temps)
                            ]
                )
            ]
        )

        return reg_mult_layout








    
    