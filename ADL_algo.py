#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithme ADL
"""

import os
import pandas as pd
import time
import plotly.express as px
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, RepeatedKFold, train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

os.getcwd()


def Selection_variables(X_disj, y, nb_max_variables):
    if len(X_disj.columns) < nb_max_variables :
        return X_disj
    else :
        selector = SelectKBest(chi2, k = nb_max_variables)
        selector.fit_transform(X_disj, y)
        X_kbest = X_disj.loc[:,selector.get_support()]
        return X_kbest


def main_algo_ADL(df, var_cible, var_exp, solv, nb_max_variables, t_ech_test, nb_splits, nb_repeats, standardisation = True):
    # ------------ A) Traitement du dataset ------------
    # Définir les var explicatives et la var cible
    X = df[var_exp]
    y = df[var_cible]
    
    # Recodage des variables quali en quanti
    X_o = pd.get_dummies(X, drop_first=True)
    
    # Selection de variables (si necessaire)
    X_ok = Selection_variables(X_o, y, nb_max_variables)
    
    # Standardisation des données (centrer-réduire)
    if standardisation == True :
        sc = StandardScaler()
        X_okok = sc.fit_transform(X_ok)
        X_okok = pd.DataFrame(X_ok)
    else : 
        X_okok = X_ok
    
    # ------------ B) Instanciation -----------
    lda = LinearDiscriminantAnalysis(solver=solv, n_components = 2)
    
    # ------------ C) Validation croisée -----------
    Valid_croisee = RepeatedKFold(n_splits = nb_splits, n_repeats = nb_repeats, random_state = 0)

    start = time.time()
    scores = cross_val_score(lda, X_okok, y, cv = Valid_croisee)
    end = time.time()
    
    Temps = end - start
    print("Taux de précision moyen en validation croisée : %.2f%%" % (scores.mean()*100.0))
    print("Durée (en seconde) = " + str(round(Temps,2)))
    
    # ------------ D) Estimation ponctuelle -----------
    
    XTrain,XTest,yTrain,yTest = train_test_split(X_okok, y, test_size= t_ech_test)

    model = lda.fit(XTrain, yTrain)
    predLda = model.predict(XTest)
    
    # Matrice de confusion
    mc = pd.crosstab(yTest,predLda)
    print("Matrice de confusion :")
    print(mc)
    
    # Structure temporaire pour affichage des coefficients
    tmp= pd.DataFrame(lda.coef_.transpose(), columns = lda.classes_, index = X.columns)
    print("Affichage des coefficients :")
    print(tmp)
    
    Aff = lda.fit(XTest, yTest).transform(XTest)
    Aff_df = pd.DataFrame(Aff, columns=["Axe1","Axe2"]) 
    Aff_df["yPred"] = predLda
    
    # Affichage
    fig = px.scatter(Aff_df, x="Axe1", y="Axe2", color='yPred')
    return fig.show()
    
    
    
    
    
 
# -------------------------- Test de main_algo_KNN() -------------------------- 
# Paramètres 
df = pd.read_csv("iris_data.csv", sep = ",")

var_cible = "species"
var_exp = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

solv = "eigen" # Solveur à utiliser : eigen, svd, lsqr 
nb_max_variables = 20
t_ech_test = 0.25 # Taille de l'echantillon de test

# Pour la cross validation :
nb_splits = 5 # Nombre de splits
nb_repeats = 20 # Nombre de repetition 

main_algo_ADL(df, var_cible, var_exp, solv, nb_max_variables, t_ech_test, nb_splits, nb_repeats, standardisation = False)

















