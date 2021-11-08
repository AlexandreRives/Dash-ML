import dash
import dash_html_components as html
import dash_core_components as dcc

#############################################################
#                   ONGLETS REGRESSION                      #
#############################################################



# Régression multiple linéaire
regmul_layout = html.Div(children=
    [
        html.P('Nombre de splits :'),
        dcc.Input(id='nb_splits', value=5, type='number', min=1, max=20, step=1),
        html.Br(),
        html.Br(),
        html.P('Nombre de répétitions :'),
        dcc.Input(id='nb_repeats', value=20, type='number', min=1, max=250, step=1),
        html.Br(),
        html.Br(),
        html.P('Nombre de variables à sélectionner :'),
        dcc.Input(id='nb_variables', value=1, type='number', min=1, max=50, step=1),
        html.Br(),
        html.Br(),
        html.Div(html.Button("Lancer l'algorithme", id='submit-regmul', n_clicks=0, className="buttonClick"), style={'textAlign': 'center', 'display': 'block'}),
        html.Div(id='analyse_regmul'),
    ], style={'margin-left': '10px', 'margin-top': '30px'}
)
    
# Arbre de décision
arbre_layout = html.Div(children=
    [
        html.P('Nombre de splits :'),
        dcc.Input(id='nb_splits', value=5, type='number', min=1, max=20, step=1),
        html.Br(),
        html.Br(),
        html.P('Nombre de répétitions :'),
        dcc.Input(id='nb_repeats', value=20, type='number', min=1, max=250, step=1),
        html.Br(),
        html.P("Choisir le nombre de feuilles :"),
        dcc.Input(id='nb_feuilles', value=5, type='number', min=5, max=50, step=1),
        html.P("Choisir le nombre de d'individus :"),
        dcc.Input(id='nb_individus', value=5, type='number', min=5, max=10000, step=1),
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