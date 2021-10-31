import dash
import dash_html_components as html
import dash_core_components as dcc

#############################################################
#                   ONGLETS REGRESSION                      #
#############################################################



# Régression multiple linéaire
regmul_layout = html.Div(children=
    [
        html.Br(),
        html.Br(),
        html.Div(html.Button("Lancer l'algorithme", id='submit-regmul', n_clicks=0, className="buttonClick"), style={'textAlign': 'center', 'display': 'block'}),
        html.Div(id='analyse_regmul'),
    ], style={'margin-left': '10px', 'margin-top': '30px'}
)
    
# Arbre de décision
arbre_layout = html.Div(children=
    [
        html.P('Choisir le nombre de feuilles :'),
        dcc.Input(id='nb_feuilles', value=1, type='number', min=1, max=10, step=1),
        html.Br(),
        html.Br(),
        html.Div(html.Button("Lancer l'algorithme", id='submit-arbre', n_clicks=0, className="buttonClick"), style={'textAlign': 'center', 'display': 'block'}),
        html.Div(id='analyse_arbre'),
    ], style={'margin-left': '10px', 'margin-top': '30px'}
)

# KNN
knn_layout = html.Div(children=
    [
        html.Br(),
        html.Br(),
        html.Div(html.Button("Lancer l'algorithme", id='submit-knn', n_clicks=0, className="buttonClick"), style={'textAlign': 'center', 'display': 'block'}),
        html.Div(id='analyse_knn'),
    ], style={'margin-left': '10px', 'margin-top': '30px'}
)