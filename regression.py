from dash import dash_table, dcc
from dash import html
from numpy.lib.function_base import select
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import RFECV

import pandas as pd
import numpy as np
import plotly.express as px
import time
import plotly.graph_objects as go
import plotly.figure_factory as ff

class Regression():
    #############################################################
    #             CONSTRUCTEUR CLASSE REGRESSION                #
    #############################################################

    def __init__(self, df, varX, varY, t_test):
        self.df = df
        self.varX = varX
        # dict_X = []
        # for i in range(0, len(self.varX)):
        #     dict_X.append(list(varX[i].values()))
        # liste_X = []
        # for i in range(0, len(dict_X)):
        #     liste_X.append(dict_X[i][0])
        self.varY = varY
        self.dfX = df[varX]
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

                        html.P("L'algorithme des K plus proches voisins vous permet de visualiser comment votre dataframe prédit vos différents individus. Vous pourrez grâce aux différents paramètres relancer l'algorithme qui prédira au mieux vos individus.", style = {'textAlign':'center'}),

                        html.Br(),
                        html.H5("Graphique de comparaison des valeurs observées et valeurs prédites", style={'textAlign':'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        dcc.Graph(figure=plot_scatter, style={'width': '30%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                        html.Br(),
                        html.Div(children=
                            [
                                html.P("Racine de l'erreur quadratique moyenne : ", style={'fontWeight':'bold', 'textAlign':'center'}),
                                html.Div(round(rmse, 2), style = {'textAlign':'center'}),
                                html.Br(),
                                html.P("Temps d'exécution de l'algorithme en validation croisée en secondes : ", style={'fontWeight':'bold', 'textAlign':'center'}),
                                html.Div(temps, style = {'textAlign':'center'})
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
        
       # ------------ A) Traitement du dataset ------------
        
        # Centrer-réduire les variables quantitatives si standardisation = Oui
        if standardisation == 'Oui' :
            sc = StandardScaler()
            X_quant = pd.DataFrame(sc.fit_transform(self.dfX_quanti))
            X_quant.columns = self.dfX_quanti.columns
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
        
        XTrain, XTest, yTrain, yTest = train_test_split(X_ok,self.dfY, test_size = self.t_test, random_state = 42)
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
        coeff = np.round(elc.coef_,4)

        #Tableau montrant les coefficients conservés après régularisation (coefficients != 0)
        coeff_penal = pd.DataFrame([coeff], columns = np.transpose(XTrain.columns))
        #coeff_penal = coeff_penal.loc[:, (coeff_penal != 0).all(axis=0)]
        
        #On transpose pour passer en colonnes
        coeff_penal = coeff_penal.T.rename(columns = {0 : "Coefficients"})
        
        #On crée la colonne Variables
        coeff_penal["Variables"] = XTrain.columns
        
        # ------------ E) Validation croisée -----------
        
        cv = RepeatedKFold(n_splits= nb_splits, n_repeats= nb_repeats, random_state=0)
        
        start = time.time()
        scores = cross_val_score(elc, X_ok, self.dfY, cv = cv, n_jobs=-2)
        end = time.time()
        
        temps = round((end - start), 2)

        # ------------ F) Visualisation -----------
        #Graphe montrant y observés points et modèle entraîné
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=yTest,
                    mode='markers',
                    name='observations'))
        fig.add_trace(go.Scatter(y=y_pred,
                    mode='lines',
                    name='prédictions'))
        
        #Graphe montrant l'évolution de l'estimateur des k validations croisées
        fig2 = go.Figure(data=go.Scatter(y=scores,
                                        mode='lines+markers',
                                        name='Scores'))
        
        #Bar graphe montrant les coefficients du modèle selon la pénalité choisie
        fig3 = go.Figure(data=[go.Bar(x=coeff_penal['Variables'], y=coeff_penal['Coefficients'])])
        fig3.update_layout(barmode='stack',
                        xaxis={'categoryorder':'total descending'})

        elastic_net_layout = html.Div(children=
            [
                html.Hr(),
                html.Br(), 
                html.Div(children=
                    [
                        html.H4("Présentation de l'algorithme de la régression ElasticNet", style={'textAlign': 'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        html.Br(),
                        html.P("L'algorithme de régression ElasticNet vous permet de visualiser comment le modèle prédit vos différents individus. Vous pourrez grâce aux différents paramètres relancer l'algorithme qui prédira au mieux vos individus.", style={'textAlign': 'center'}),
                        html.P("Le paramètre l1 ratio, compris entre 0 et 1 vous permet d'orienter la pénalité vers Ridge, Lasso ou entre les deux", style={'textAlign': 'center'}),
                        html.P("Le paramètre alpha est un multiplicateur de la pénalité.", style={'textAlign': 'center'}),
                        html.Br(),
                        html.H5("Graphique de comparaison des valeurs observées et valeurs prédites", style={'textAlign':'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        dcc.Graph(figure=fig, style={'width': '70%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                        html.Br(),
                        html.H5("Evolution des estimateurs de la validation croisée", style={'textAlign':'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        dcc.Graph(figure = fig2, style={'width': '70%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                        html.H5("Paramètres du modèle", style={'textAlign':'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        html.Br(),
                        html.P("Le graphe ci-dessous présente les paramètres du modèle. Les variables sont sélectionnées selon les hyperparamètres l1_ratio et alpha", style={'fontWeight':'bold','textAlign':'center'}),
                        dcc.Graph(figure = fig3, style={'width': '70%', 'margin-left':'auto', 'margin-right':'auto'}),
                        html.Br(),
                        html.Div(children=
                            [
                                html.P("Racine de l'erreur quadratique moyenne : " , style={'fontWeight':'bold', 'textAlign':'center'}),
                                html.Div(round(RMSE, 2), style={'textAlign':'center'}),
                                html.Br(),
                                html.P("Coefficient de détermination R² : ", style={'fontWeight':'bold', 'textAlign':'center'}),
                                html.Div(round(perf, 2), style={'textAlign':'center'}),
                                html.Br(),
                                html.P("Temps d'exécution de l'algorithme en validation croisée en secondes : ", style={'fontWeight':'bold', 'textAlign':'center'}),
                                html.Div(temps, style={'textAlign':'center'})
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

        # ------------ E) Sélection de variables (pas très concluante) -----------
        estimator = RFECV(RegMul, step=1, cv=5)
        selector = estimator.fit(XTrain, yTrain)
        ypred = selector.predict(XTest)

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
                html.P("L'algorithme de la régression linéaire multiple vous permet de visualiser comment votre dataframe prédit vos différents individus. Vous pourrez grâce aux différents paramètres relancer l'algorithme qui prédira au mieux vos individus.", style = {'textAlign':'center'}),
                html.Br(),
                html.Div(children=
                    [
                        html.H5("Graphique de comparaison des valeurs observées et valeurs prédites", style={'textAlign':'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        dcc.Graph(figure=plot_scatter, style={'width': '30%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                    ]
                ),
                html.Div(children=
                            [
                                html.P("Racine de l'erreur quadratique moyenne : ", style={'fontWeight':'bold', 'textAlign':'center'}),
                                html.Div(round(RMSE, 2), style = {'textAlign':'center'}),
                                html.Br(),
                                html.P("Coefficient de détermination R² : ", style={'fontWeight':'bold', 'textAlign':'center'}),
                                html.Div(round(perf, 2), style = {'textAlign':'center'}),
                                html.Br(),
                                html.P("Temps d'execution de l'algorithme en validation croisée en seconde : ", style={'fontWeight':'bold', 'textAlign':'center'}),
                                html.Div(temps, style = {'textAlign':'center'})
                            ]
                )
            ]
        )

        return reg_mult_layout








    
    