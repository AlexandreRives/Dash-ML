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
* En rouge, il s’agit des algorithmes de classification, à utiliser lorsque la variable cible est catégorielle :
  * Arbre de décision
  * Analyse discriminante linéaire
  * Régression logistique
* En bleu, il s’agit des algorithmes de régression, à utiliser lorsque la variable cible est quantitative :
  * K plus proches voisins
  * ElasticNet
  * Régression linéaire multiple

Chacun de ces algorithmes possède des paramètres spécifiques que l’utilisateur peut faire varier à sa guise.

