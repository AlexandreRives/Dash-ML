import numpy as np
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import RepeatedKFold

import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly import tools
import plotly.figure_factory as ff
    
import dash
from dash import dcc
from dash import html
    
import warnings
warnings.filterwarnings('ignore')
import time


class algo_classification():
    
    def __init__(self, df, varX, varY, t_test):
        self.df = df
        self.varX = varX
        self.varY = varY
        self.dfX = df[varX]
        self.dfX_quanti = self.dfX.select_dtypes(include=[np.number])
        self.dfX_quali = self.dfX.select_dtypes(exclude=[np.number])
        self.dfY = df[varY]
        self.t_test = t_test


    def Regression_log(self, nb_splits, n_repeats, standardisation, iterations, l1_ratio, C, cores = 1):
        ''' Options
        
        nb_splits : nombre de scission du dataset pour la validation croisée ====> nb positif entier
        
        n_repeats : nombre de répétition de la validation croisée ====> nb positif entier
        
        standardisation : centrer-réduire les données ====> 'oui' ou 'non'
        
        iterations : itération max dans l'instanciation du modèle ====> nb positif entier
        
        l1_ratio : curseur entre la pénalisation Ridge et Lasso ====> 0 (l2) <= l1_ratio <= 1 (l1)
       
        C : hyperparamètre permettant d'allouer une plus forte pénalité aux données (C faible), ie on fait peu confiance au caractère généralisable des données, ou à la complexité du modèle ie on fait confiance aux données (C grand) 
        
        cores ====> nb de coeurs pour le calcul parallèle. Si = 1, pas de parallélisation, si = -1, tous les coeurs sont utilisés, si = -2, tous les coeurs sauf 1 sont utilisés, etc.'''


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
            X_qual = pd.get_dummies(self.dfX_quali, 
                                    drop_first=True)
            # Concatener X_qual et X_quant 
            X_ok = pd.concat([X_quant,X_qual], axis = 1)

        # ------------ B) Instanciation du modèle -----------
        
        if len(np.unique(self.dfY))==2 : #si la variable cible a 2 modalités : régression logistique binaire
            Type_classe = 'binaire'
            #instanciation du modèle binaire
            lr = LogisticRegression(solver = 'saga', 
                                    max_iter = iterations, 
                                    penalty = 'elasticnet', 
                                    l1_ratio = l1_ratio, 
                                    C=C)
        
        #si la variable cible a plus de 2 modalités : régression logistique multiclasse
        elif len(np.unique(self.dfY))==2 :
            Type_classe = 'multiclasse'
            #instanciation du modèle multiclasse
            lr = LogisticRegression(solver = 'saga', 
                                    max_iter = iterations, 
                                    penalty = 'elasticnet', 
                                    l1_ratio = l1_ratio, 
                                    multi_class = 'ovr', 
                                    C=C, 
                                    n_jobs= cores) 
        
        #si la variable a moins de 2 modalités : renvoie une erreur
        else:
            raise ValueError('La variable cible ne possède pas assez de modalités. Minimum : 2')

        # ------------ C) Split en échantillons d'apprentissage et de test -----------

        #Scission du dataset en échantillons d'apprentissage et test
        XTrain, XTest, yTrain, yTest = train_test_split(X_ok, 
                                                        self.dfY, 
                                                        test_size = self.t_test, 
                                                        random_state = 1)

        #Entraînement
        lr.fit(XTrain, yTrain)

        #Prédiction
        y_pred = lr.predict(XTest)

        # ------------ D) Performance -----------
        
        #Matrice de confusion
        mc = confusion_matrix(yTest, 
                            y_pred, 
                            labels=lr.classes_)
        
        #Accuracy
        accuracy = accuracy_score(yTest, y_pred)
        
        #Coefficients du modèle
        coeff = lr.coef_ 
        
        #taux de reconnaissane :
        tx_reco = (sum(np.diag(mc)) / len(XTest)) * 100

        # ------------ E) Validation croisée -----------

        crossv  = RepeatedKFold(n_splits = nb_splits, 
                                n_repeats = n_repeats, 
                                random_state=0)
        
        start = time.time()
        scores = cross_val_score(lr, 
                                X_ok, 
                                self.dfY, 
                                cv= crossv, 
                                scoring = 'accuracy') 
        end = time.time()
        
        temps = round((end - start), 2)

        # ------------ F) Visualisation -----------
        
        z = mc

        #Récupération des labels dans l'attribut .classes_
        labels = lr.classes_.reshape(1,len(lr.classes_))
        
        label_obs = []
        for j in range(len(lr.classes_)):
            label_obs.append(labels[0,j])
       
        #Création des labels prédiction
        label_pred = [label + " pred" for label in label_obs]
               
        #Matrice de confusion
        fig = ff.create_annotated_heatmap(z, 
                                        x=label_pred, 
                                        y=label_obs)

        fig.update_layout(title_text='Matrice de confusion')   

        #Estimateurs de la validation croisée
        fig2 = go.Figure(data=go.Scatter(y=scores,
                                        mode='lines+markers',
                                        name='Scores'))
        
        fig2.update_layout(title='Estimateurs de la validation croisée')
        
        regression_logistic_layout = html.Div(children=
            [
                html.Hr(),
                html.Br(), 
                html.Div(children=
                    [
                        html.H4("Présentation de l'algorithme de la régression logistique", style={'textAlign': 'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        html.Br(),
                        html.P("L'algorithme de régression logistique vous permet de visualiser comment le modèle classe vos différents individus. Vous pourrez grâce aux différents paramètres relancer l'algorithme qui prédira au mieux vos individus."),
                        html.Br(),
                        html.P("Le paramètre l1 ratio, compris entre 0 et 1 vous permet d'orienter la pénalité vers Ridge, Lasso ou entre les deux"),
                        html.P("Le paramètre alpha est "),
                        html.Br(),
                        html.P("Enfin, nous vous affichons la matrice de confusion avec la métrique suivante  : le taux d'erreur."),
                        html.H5("Matrice de confusion", style={'textAlign':'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        dcc.Graph(figure = fig, style={'width': '50%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                        html.Br(),
                        html.H5("Evolution des estimateurs de la validation croisée", style={'textAlign':'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        dcc.Graph(figure = fig2, style={'width': '70%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                        html.Br(),
                        html.Div(children=
                            [
                                html.Span("Accuracy : ", style={'fontWeight':'bold'}),
                                html.Div(round(accuracy, 2)),
                                html.Br(),
                                html.Span("Taux de reconnaissance : ", style={'fontWeight':'bold'}),
                                html.Div(round(tx_reco, 2)),
                                html.Br(),
                                html.Span("Coefficients du modèle : ", style={'fontWeight':'bold'}),
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

        return regression_logistic_layout

        #predict_proba ?
        #boucle for pour cross val avec différent scoring pour comparaison ?