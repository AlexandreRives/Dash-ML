    
from typing import _T_co
import numpy as np
from numpy import absolute
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objects as go
from plotly import tools
    
import dash
from dash import dcc, dash_table
from dash import html
    
import warnings
warnings.filterwarnings('ignore')    
import time

class Regression():

    def __init__(self, df, varX, varY, t_test):
        self.df = df
        self.varX = varX
        self.varY = varY
        self.dfX = df[varX]
        self.dfX_quanti = self.dfX.select_dtypes(include=[np.number])
        self.dfX_quali = self.dfX.select_dtypes(exclude=[np.number])
        self.dfY = df[varY]
        self.t_test = t_test
    
    
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
        
        XTrain, XTest, yTrain, yTest = train_test_split(X_ok, self.dfY, test_size = self.t_test, random_state = 1)
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
        scores = cross_val_score(elc, X_ok, self.dfY, cv = cv, n_jobs=-1)
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
        
        fig2.update_layout(title='Validation Croisée selon le métrique Mean absolute error regression loss')
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
                                html.Span("Coefficients du modèles : ", style={'fontWeight':'bold'}),
                                html.Div(round(coeff, 2)),
                                html.Br(),
                                html.Span("Temps d'exécution de l'algorithme en validation croisée en secondes : ", style={'fontWeight':'bold'}),
                                html.Div(temps)
                            ]
                        )

                    ]
                ),   
            ]
        )

        return elastic_net_layout
        
#Rajouter l'affichage des coeffs pour montrer la réduction de variable permise avec elasticnet en jouant sur le l1_ratio
        