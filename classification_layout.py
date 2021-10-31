import dash
import dash_html_components as html
import dash_core_components as dcc

#############################################################
#                   ONGLETS CLASSIFICATION                  #
#############################################################

# SVM
SVM_layout = html.Div(children=
    [
        html.Br(),
        html.Br(),
        html.Div(html.Button("Lancer l'algorithme", id='submit-svm', n_clicks=0, className="buttonClick"), style={'textAlign': 'center', 'display': 'block'}),
        html.Div(id='analyse_svm'),
    ], style={'margin-left': '10px', 'margin-top': '30px'}
)

# Analyse discriminante linéaire
adl_layout = html.Div(children=
    [
        html.P("Taille de l'échantillon test :"),
        dcc.Input(id='t_ech_test', value=0.3, type='number', min=0, max=1, step=0.1),
        html.Br(),
        html.P("Solveur :"),
        dcc.Dropdown(id='solv', options=[{'label' : 'svd', 'value' : 'svd'}, {'label' : 'logr', 'value' : 'logr'}, {'label' : 'eigen', 'value' : 'eigen'}], value='svd', style={'width': '300px'}),
        html.Br(),
        html.Br(),
        html.P('Nombre de splits :'),
        dcc.Input(id='nb_splits', value=10, type='number', min=1, max=20, step=1),
        html.Br(),
        html.Br(),
        html.P('Nombre de répétitions :'),
        dcc.Input(id='nb_repeats', value=50, type='number', min=1, max=100, step=10),
        html.Br(),
        html.Br(),
        html.Div(html.Button("Lancer l'algorithme", id='submit-adl', n_clicks=0, className="buttonClick"), style={'textAlign': 'center', 'display': 'block'}),
        html.Div(id='analyse_adl'),
    ], style={'margin-left': '10px', 'margin-top': '30px'}
)

# Régression logistique
reglog_layout = html.Div(children=
    [
        html.Br(),
        html.Br(),
        html.Div(html.Button("Lancer l'algorithme", id='submit-reglog', n_clicks=0, className="buttonClick"), style={'textAlign': 'center', 'display': 'block'}),
        html.Div(id='analyse_reglog'),
    ], style={'margin-left': '10px', 'margin-top': '30px'}
)

# CAH
# cah_layout = html.Div(children=
#     [
#         html.Br(),
#         html.Br(),
#         html.Div(html.Button("Lancer l'algorithme", id='submit-cah', n_clicks=0, className="buttonClick"), style={'textAlign': 'center', 'display': 'block'}),
#         html.Div(id='analyse_cah'),
#     ], style={'margin-left': '10px', 'margin-top': '30px'}
# )


