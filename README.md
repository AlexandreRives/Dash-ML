# Projet Machine Learning sous Python avec Dash - Master 2 SISE

* Frintz Elisa
* Madi Corodji Jacky
* Rives Alexandre

## Contexte du projet

Dans le cadre de notre cours de __machine learning sous python__, nous avons dû créer une __interface d’analyse de données par apprentissage supervisé__. 
L’objectif de notre application est qu’un utilisateur (sans réelles connaissances) puisse appliquer une ou plusieurs méthodes de machine learning au jeu de données de son choix. Il pourra ainsi comparer leurs performances et déterminer quel est l’algorithme qui s’applique au mieux à son jeu de données.

## Guide d’utilisation
1. Télécharger le dossier .zip à partir du dépôt github

2. Vérifier que toutes les librairies utilisées sont correctement installées :
base64, io, dash, pandas, numpy, cchardet, sklearn, plotly, os, time, dash bootstrap components, dash html components, dash core components

3. Dans un terminal, se déplacer dans le dossier du projet et exécuter le fichier "app.py" à l’aide des commandes : 
```
cd _chemin_
python app.py
```
Vous devez obtenir l’interface suivante :
![img1](https://user-images.githubusercontent.com/65174929/162714200-4629e65e-c33f-45a2-8ef9-6fbe1e055ba9.png)

4. Importation du fichier (au format .csv uniquement) et choix des paramètres généraux relatifs à l’ensemble des algorithmes d’apprentissage suppervisé
(la variable cible, les variables explicatives, la taille de l’echantillon de test et le centrage-réduction des variables explicatives).

Voici un exemple avec le jeu de données iris.csv :
![img2](https://user-images.githubusercontent.com/65174929/162714982-03957859-8ecf-43a3-8917-bd55dcf88c0a.png)

5. Choix de l’algorithme à utiliser :
* En rouge, il s’agit des __algorithmes de classification__, à utiliser lorsque la __variable cible est catégorielle__ :
  * Arbre de décision
  * Analyse discriminante linéaire
  * Régression logistique
* En bleu, il s’agit des __algorithmes de régression__, à utiliser lorsque la __variable cible est quantitative__ :
  * K plus proches voisins
  * ElasticNet
  * Régression linéaire multiple

Chacun de ces algorithmes possède des paramètres spécifiques que l’utilisateur peut faire varier à sa guise.

6. En fonction de l’algorithme choisi, les paramètres ne sont pas les mêmes. Cependant, leurs résultats restent comparables entre eux. L’utilisateur peut alors jouer avec les paramètres pour trouver la meilleure combinaison possible. 

Par exemple, l’algorithme de classification "arbre de décision" ressemble à cela :
![img3](https://user-images.githubusercontent.com/65174929/162715836-9501bd31-c2b4-4146-859c-c51d0ee35cea.png)
![img4](https://user-images.githubusercontent.com/65174929/162715861-e171479e-61c0-4403-8467-d676331a3f95.png)


