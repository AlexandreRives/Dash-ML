#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fichier Algo de classification
"""

from dash import dcc
from dash import html
from sklearn.cluster import KMeans
from sklearn import cluster, metrics
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report

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

class classification ():
    
    #############################################################
    #            CONSTRUCTEUR CLASSE CLASSIFICATION             #
    #############################################################

    def __init__(self, df, varX, varY, t_ech_test):
        self.df = df
        self.varX = varX
        self.varY = varY
        self.dfX = df[varX]
        self.dfX_quanti = self.dfX.select_dtypes(include=[np.number])
        self.dfX_quali = self.dfX.select_dtypes(exclude=[np.number])
        self.dfY = df[varY]
        self.t_ech_test = t_ech_test

    #############################################################
    #            ANALYSE DISCRIMINANTE LINEAIRE                 #
    #############################################################
    
    def algo_ALD(self, solv, nb_splits, nb_repeats, standardisation = True):
        # ------------ A) Traitement du dataset ------------
        # Recoder les variables qualitatives
        X_qual = pd.get_dummies(self.dfX_quali, drop_first=True)
        
        # Centrer-réduire les variables quantitatives si standardisation = True
        if standardisation == True :
            sc = StandardScaler()
            X_quant = pd.DataFrame(sc.fit_transform(self.dfX_quanti))
        else : 
            X_quant = self.dfX_quanti
            
        # Concatener X_qual et X_quant 
        X_ok = pd.concat([X_quant,X_qual], axis = 1)
        
        # ------------ B) Instanciation -----------
        lda = LinearDiscriminantAnalysis(solver=solv, n_components = 2)
        
        # ------------ C) Validation croisée -----------
        Valid_croisee = RepeatedKFold(n_splits = nb_splits, n_repeats = nb_repeats, random_state = 0)
        
        start = time.time()
        scores = cross_val_score(lda, X_ok, self.dfY, cv = Valid_croisee)
        end = time.time()
        
        Temps = end - start
        print("Taux de précision moyen en validation croisée : %.2f%%" % (scores.mean()*100.0))
        print("Durée (en seconde) = " + str(round(Temps,2)))
        
        # ------------ D) Estimation ponctuelle -----------
        XTrain,XTest,yTrain,yTest = train_test_split(X_ok, self.dfY, test_size = self.t_ech_test, random_state = 0)
        
        model = lda.fit(XTrain, yTrain)
        predLda = model.predict(XTest)
        
        # Matrice de confusion
        mc = pd.crosstab(yTest,predLda)
        
        print("Matrice de confusion :")
        print(mc)
        
        # Structure temporaire pour affichage des coefficients
        tmp= pd.DataFrame(lda.coef_.transpose(), columns = lda.classes_, index = X_ok.columns)
        print("Affichage des coefficients :")
        print(tmp)
        
        Aff = lda.fit(XTest, yTest).transform(XTest)
        Aff_df = pd.DataFrame(Aff, columns=["Axe1","Axe2"]) 
        Aff_df["yPred"] = predLda
        
        # Affichage
        fig = px.scatter(Aff_df, x="Axe1", y="Axe2", color='yPred')
        
        return fig.show()
    
    #############################################################
    #                     ARBRE DE DECISION                     #
    #############################################################
    
    def algo_Arbre(self, nb_feuilles, nb_splits, nb_individus, nb_repeats, standardisation = True):
        # ------------ A) Traitement du dataset ------------
        # Recoder les variables qualitatives
        X_qual = pd.get_dummies(self.dfX_quali, drop_first=True)
        
        # Centrer-réduire les variables quantitatives si standardisation = True
        if standardisation == True :
            sc = StandardScaler()
            X_quant = pd.DataFrame(sc.fit_transform(self.dfX_quanti))
        else : 
            X_quant = self.dfX_quanti
            
        # Concatener X_qual et X_quant 
        X_ok = pd.concat([X_quant,X_qual], axis = 1)
        
        # ------------ B) Instanciation -----------
        arbre = DecisionTreeClassifier(max_depth = nb_feuilles, max_leaf_nodes = nb_individus, random_state = 5)
       
        # ------------ C) Validation croisée -----------
        Valid_croisee = RepeatedKFold(n_splits = nb_splits, n_repeats = nb_repeats, random_state = 0)
        
        start = time.time()
        scores = cross_val_score(arbre, X_ok, self.dfY, cv = Valid_croisee)
        end = time.time()
       
        Temps = end - start
        print("Taux de précision moyen en validation croisée : %.2f%%" % (scores.mean()*100.0))
        print("Durée (en seconde) = " + str(round(Temps,2)))
        
        # PLOT DE LA VALIDATION CROISEE
        K_range = []
        for i in range(1, (nb_repeats*nb_splits)+1):
            K_range.append(i)
        scores_table = pd.DataFrame(data={'K_range': K_range, 'scores': scores})
        plot_line = px.line(scores_table, x="K_range", y="scores")
        
        # ------------ D) Estimation ponctuelle -----------
        XTrain,XTest,yTrain,yTest = train_test_split(X_ok, self.dfY, test_size = self.t_ech_test, random_state = 0)
       
        model = arbre.fit(XTrain, yTrain)
        ypred = model.predict(XTest)
       
        # Matrice de confusion
        mc = pd.crosstab(yTest,ypred)
        print("Matrice de confusion :")
        print(mc)
       
       
       

        
       






