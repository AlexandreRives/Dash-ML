from dash import html
from dash import dash_table, dcc
from numpy.core import numeric
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, RepeatedKFold, train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

import pandas as pd
import numpy as np
import plotly.express as px
import os
import pandas as pd
import time
import plotly.graph_objs as go
import plotly.figure_factory as ff

class Classification():

    #############################################################
    #            CONSTRUCTEUR CLASSE CLASSIFICATION             #
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
    #                     ARBRE DE DECISION                     #
    #############################################################
    
    def algo_arbre(self, nb_feuilles, nb_individus, nb_splits, nb_repeats, standardisation):
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
        
        # ------------ B) Instanciation -----------
        arbre = DecisionTreeClassifier(max_depth = nb_feuilles, max_leaf_nodes = nb_individus, random_state = 5)
       
        # ------------ C) Validation croisée -----------
        Valid_croisee = RepeatedKFold(n_splits = nb_splits, n_repeats = nb_repeats, random_state = 0)
        
        start = time.time()
        scores = cross_val_score(arbre, X_ok, self.dfY, cv = Valid_croisee)
        end = time.time()
       
        temps = round((end - start), 2)
    
        # ------------ D) Estimation ponctuelle -----------
        XTrain,XTest,yTrain,yTest = train_test_split(X_ok, self.dfY, test_size = self.t_test, random_state = 42)
       
        model = arbre.fit(XTrain, yTrain)
        ypred = model.predict(XTest)

        # IMPORTANCE DE VARIABLES #
        importance_var = {"Variables":X_ok.columns, "Coefficients":arbre.feature_importances_}
        importance_var = pd.DataFrame(importance_var).sort_values(by="Coefficients", ascending=False)

        # METRIQUES #
        #Matrice de confusion
        mc = confusion_matrix(yTest, ypred, labels=model.classes_)
        tx_reconaissance = (sum(np.diag(mc)) / len(XTest)) * 100  

        # ------------ F) Visualisation -----------
        
        #Récupération des labels dans l'attribut .classes_
        labels = model.classes_.reshape(1,len(model.classes_))
        
        label_obs = []
        for j in range(len(model.classes_)):
            label_obs.append(labels[0,j])

        #Transtypage de int en string
        label_obs = [str(numeric_string) for numeric_string in label_obs]
       
        #Création des labels prédiction
        label_pred = [label + " pred" for label in label_obs]
               
        #Matrice de confusion
        fig = ff.create_annotated_heatmap(z = mc, 
                                        x=label_pred, 
                                        y=label_obs, colorscale='blues') 

        #Estimateurs de la validation croisée
        fig2 = go.Figure(data=go.Scatter(y=scores,
                                        mode='lines+markers',
                                        name='Scores'))

        #Bar graphe montrant les coefficients du modèle selon la pénalité choisie
        fig3 = go.Figure(data=[go.Bar(x=importance_var['Variables'], y=importance_var['Coefficients'])])
        fig3.update_layout(barmode='stack',
                        xaxis={'categoryorder':'total descending'})

        # AFFICHAGE DU LAYOUT #
        arbre_algo_layout = html.Div(children=
            [
                html.Hr(),
                html.Br(), 
                html.Div(children=
                    [
                        html.H4("Présentation de l'algorithme de l'arbre de décision", style={'textAlign': 'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        html.Br(),

                        html.P("L'algorithme de l'arbre de décision vous permet de visualiser comment votre dataframe classe vos différents individus. Vous pourrez grâce aux différents paramètres relancer l'algorithme qui classifiera au mieux vos individus.", style = {'textAlign':'center'}),

                        html.H5("Tableau d'importance des variables", style={'textAlign':'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        html.Br(),

                        html.P("Nous effectuons pour vous une analyse préliminaire qui vous permet de choisir le nombre de variable à retenir. Ci-dessous le tableau de l'importance des variables : ", style = {'textAlign':'center'}),

                        dcc.Graph(figure = fig3, style={'width': '40%', 'margin-left':'auto', 'margin-right':'auto', 'margin-top':'auto'}),
                        html.H5("Matrice de confusion", style={'textAlign':'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        dcc.Graph(figure = fig, style={'width': '40%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                        html.P("Taux de reconnaissance en % en estimation ponctuelle : ", style={'fontWeight':'bold', 'textAlign':'center'}),
                        html.Div(round(tx_reconaissance, 2), style = {'textAlign':'center'}),
                        html.Br(),
                        html.H5("Graphe de l'évolution du taux de reconnaissance en validation croisée", style={'textAlign':'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        dcc.Graph(figure=fig2, style={'width': '70%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                        html.Div(children=
                            [
                                html.P("Taux de reconnaissance en % en validation croisée : ", style={'fontWeight':'bold', 'textAlign':'center'}),
                                html.Div(round((scores.mean()*100.0), 2), style = {'textAlign':'center'}),
                                html.Br(),
                                html.P("Temps d'exécution de l'algorithme en validation croisée en secondes : ", style={'fontWeight':'bold', 'textAlign':'center'}),
                                html.Div(temps, style = {'textAlign':'center'})
                            ]
                        )

                    ]
                ),   
            ]
        )

        return arbre_algo_layout

    #############################################################
    #              ANALYSE DISCRIMINANTE LINEAIRE               #
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
        #lda = LinearDiscriminantAnalysis(solver=solv, n_components = 2)
        lda = LinearDiscriminantAnalysis(solver=solv)
        # ------------ C) Validation croisée -----------
        Valid_croisee = RepeatedKFold(n_splits = nb_splits, n_repeats = nb_repeats, random_state = 0)

        start = time.time()
        scores = cross_val_score(lda, X_ok, self.dfY, cv = Valid_croisee)
        end = time.time()

        temps = round((end - start), 2)        
        
        # ------------ D) Estimation ponctuelle -----------
        XTrain,XTest,yTrain,yTest = train_test_split(X_ok, self.dfY, test_size = self.t_test, random_state = 42)
        
        model = lda.fit(XTrain, yTrain)
        predLda = model.predict(XTest)
        
        # Structure temporaire pour affichage des coefficients
        #tmp = pd.DataFrame(lda.coef_.transpose(), columns = lda.classes_, index = X_ok.columns)
        
        #Aff = lda.fit(XTest, yTest).transform(XTest)
        #Aff_df = pd.DataFrame(Aff, columns=["Axe1","Axe2"]) 
        #Aff_df["yPred"] = predLda
        
        # METRIQUES #
        mc = confusion_matrix(yTest, predLda, labels=model.classes_)
        tx_reconaissance = (sum(np.diag(mc)) / len(XTest)) * 100

        # ------------ F) Visualisation -----------
        
        #Récupération des labels dans l'attribut .classes_
        labels = lda.classes_.reshape(1,len(lda.classes_))
        
        label_obs = []
        for j in range(len(lda.classes_)):
            label_obs.append(labels[0,j])

        #Transtypage de int en string
        label_obs = [str(numeric_string) for numeric_string in label_obs]
       
        #Création des labels prédiction
        label_pred = [label + " pred" for label in label_obs]

        newlist = [x for x in scores if np.isnan(x) == False]
        print(np.mean(newlist))
               
        #Matrice de confusion
        fig = ff.create_annotated_heatmap(z = mc, 
                                        x=label_pred, 
                                        y=label_obs, colorscale='blues') 

        #Estimateurs de la validation croisée
        fig2 = go.Figure(data=go.Scatter(y=scores,
                                        mode='lines+markers',
                                        name='Scores'))

        #Visualisation ponctuelle
        #fig3 = px.scatter(Aff_df, x="Axe1", y="Axe2", color='yPred')

        # AFFICHAGE DU LAYOUT #
        adl_algo_layout = html.Div(children=
            [
                html.Hr(),
                html.Br(), 
                html.Div(children=
                    [
                        html.H4("Présentation de l'algorithme de l'analyse discriminante linéaire", style={'textAlign': 'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        html.Br(),
                        html.P("L'algorithme de l'analyse discriminante linéaire vous permet de visualiser comment votre dataframe classe vos différents individus. Vous pourrez grâce à la sélection de feuilles et du nombre maximum d'individus par feuille relancer l'algorithme qui classifiera au mieux vos individus.", style = {'textAlign':'center'}),
                        html.Br(),
                        html.H5("Matrice de confusion", style={'textAlign':'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        dcc.Graph(figure=fig, style={'width': '40%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                        html.P("Taux de reconnaissance en % en estimation ponctuelle : ", style={'fontWeight':'bold', 'textAlign':'center'}),
                        html.Div(round(tx_reconaissance, 2), style = {'textAlign':'center'}),
                        html.Br(),
                        html.H5("Graphe de l'évolution du taux de reconnaissance en validation croisée", style={'textAlign':'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        dcc.Graph(figure=fig2, style={'width': '70%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                        #html.H5("Visualisation ponctuelle", style={'textAlign':'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        #dcc.Graph(figure=fig3, style={'width': '70%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                        html.Br(),
                        html.Div(children=
                            [

                                html.P("Taux de reconaissance en % en validation croisée : ", style={'fontWeight':'bold', 'textAlign':'center'}),
                                html.Div(round(np.mean(newlist), 2), style={'textAlign': 'center'}),
                                html.Br(),
                                html.P("Temps d'exécution de l'algorithme en validation croisée en secondes : ", style={'fontWeight':'bold', 'textAlign':'center'}),
                                html.Div(temps, style = {'textAlign':'center'})
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
    
    def Regression_log(self, nb_splits, nb_repeats, standardisation, iterations, l1_ratio, C):
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
            #instanciation du modèle binaire
            lr = LogisticRegression(solver = 'saga', 
                                    max_iter = iterations, 
                                    penalty = 'elasticnet', 
                                    l1_ratio = l1_ratio, 
                                    C=C)
        
        #si la variable cible a plus de 2 modalités : régression logistique multiclasse
        else:
            #instanciation du modèle multiclasse
            lr = LogisticRegression(solver = 'saga', 
                                    max_iter = iterations, 
                                    penalty = 'elasticnet', 
                                    l1_ratio = l1_ratio, 
                                    multi_class = 'ovr', 
                                    C=C) 
        
        #si la variable a moins de 2 modalités : renvoie une erreur
        # else:
        #     raise ValueError('La variable cible ne possède pas assez de modalités. Minimum : 2')

        # ------------ C) Split en échantillons d'apprentissage et de test -----------

        #Scission du dataset en échantillons d'apprentissage et test
        XTrain, XTest, yTrain, yTest = train_test_split(X_ok, 
                                                        self.dfY, 
                                                        test_size = self.t_test, 
                                                        random_state = 42)

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
        #coeff = lr.coef_ 
        
        #taux de reconnaissane :
        tx_reco = (sum(np.diag(mc)) / len(XTest)) * 100

        # ------------ E) Validation croisée -----------

        crossv  = RepeatedKFold(n_splits = nb_splits, 
                                n_repeats = nb_repeats, 
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
        
        #Récupération des labels dans l'attribut .classes_
        labels = lr.classes_.reshape(1,len(lr.classes_))
        
        label_obs = []
        for j in range(len(lr.classes_)):
            label_obs.append(labels[0,j])
        
        #Transtypage de int en string
        label_obs = [str(numeric_string) for numeric_string in label_obs]

        #Création des labels prédiction
        label_pred = [label + " pred" for label in label_obs]
        
        #Matrice de confusion
        fig = ff.create_annotated_heatmap(z = mc, 
                                        x=label_pred, 
                                        y=label_obs, colorscale='blues') 

        #Estimateurs de la validation croisée
        fig2 = go.Figure(data=go.Scatter(y=scores,
                                        mode='lines+markers',
                                        name='Scores'))
        
        regression_logistic_layout = html.Div(children=
            [
                html.Hr(),
                html.Br(), 
                html.Div(children=
                    [
                        html.H4("Présentation de l'algorithme de la régression logistique", style={'textAlign': 'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        html.Br(),
                        html.P("L'algorithme de régression logistique vous permet de visualiser comment le modèle classe vos différents individus. Vous pourrez grâce aux différents paramètres relancer l'algorithme qui prédira au mieux vos individus.", style = {'textAlign':'center'}),
                        html.Br(),
                        html.H5("Matrice de confusion", style={'textAlign':'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        dcc.Graph(figure = fig, style={'width': '40%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                        html.P("Taux de reconnaissance en estimation ponctuelle : ", style={'fontWeight':'bold', 'textAlign':'center'}),
                        html.Div(round(tx_reco, 2), style = {'textAlign':'center'}),
                        html.Br(),
                        html.H5("Evolution des estimateurs de la validation croisée", style={'textAlign':'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
                        dcc.Graph(figure = fig2, style={'width': '70%', 'display':'block', 'margin-left':'auto', 'margin-right':'auto'}),
                        html.Br(),
                        html.Div(children=
                            [
                                html.P("Taux de reconnaissance en % en validation croisée : ", style={'fontWeight':'bold', 'textAlign':'center'}),
                                html.Div(round((scores.mean()*100.0), 2), style = {'textAlign':'center'}),
                                html.Br(),
                                html.P("Temps d'exécution de l'algorithme en validation croisée en secondes : ", style={'fontWeight':'bold', 'textAlign':'center'}),
                                html.Div(temps, style = {'textAlign':'center'}),
                            ]
                        )

                    ]
                ),   
            ]
        )

        return regression_logistic_layout
