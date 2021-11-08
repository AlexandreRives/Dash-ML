#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithme KNN
"""

import os
import pandas as pd
import time
import plotly.express as px
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, RepeatedKFold, train_test_split


os.getcwd()


def Selection_variables(X_disj, y, nb_max_variables):
    if len(X_disj.columns) < nb_max_variables :
        return X_disj
    else :
        selector = SelectKBest(chi2, k = nb_max_variables)
        selector.fit_transform(X_disj, y)
        X_kbest = X_disj.loc[:,selector.get_support()]
        return X_kbest

def main_algo_KNN(df, var_cible, var_exp, K, nb_max_variables, t_ech_test, nb_splits, nb_repeats, standardisation = True):
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
    KNN = KNeighborsRegressor(K)
    
    # ------------ C) Validation croisée -----------
    Valid_croisee = RepeatedKFold(n_splits = nb_splits, n_repeats = nb_repeats, random_state = 0)
    
    start = time.time()
    scores = cross_val_score(KNN, X_okok, y, cv = Valid_croisee)
    end = time.time()

    Temps = end - start
    
    print("Durée (en seconde) = " + str(round(Temps,2)))
    print("Score moyen en validation croisée = " + str(round(scores.mean(),2)))
    

    # ------------ D) Estimation ponctuelle -----------
    XTrain,XTest,yTrain,yTest = train_test_split(X_okok, y, test_size= t_ech_test)
    
    model = KNN.fit(XTrain, yTrain)
    y_pred = model.predict(XTest)
    
    scores_table = pd.DataFrame(data={'yTest': yTest, 'y_pred': y_pred})
    
    plot_scatter = px.scatter(scores_table, x="yTest", y="y_pred")
    
    return plot_scatter.show()
    
# -------------------------- Test de main_algo_KNN() -------------------------- 
# Paramètres 
df = pd.read_csv("student-mat.csv", sep = ";")

var_cible = "G3"
var_exp = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
       'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
       'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
       'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
       'Walc', 'health', 'absences']

K = 7 # Nombre de voisins
nb_max_variables = 20
t_ech_test = 0.25 # Taille de l'echantillon de test

# Pour la cross validation :
nb_splits = 5 # Nombre de splits
nb_repeats = 20 # Nombre de repetition 

main_algo_KNN(df, var_cible, var_exp, K, nb_max_variables, t_ech_test, nb_splits, nb_repeats, standardisation = False)








