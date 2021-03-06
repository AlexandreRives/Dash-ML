import dash
import dash_html_components as html
import dash_core_components as dcc

#############################################################
#                   ONGLETS REGRESSION                      #
#############################################################



# Régression multiple linéaire
regmul_layout = html.Div(children=[html.Div(children=
    [
        html.H6(children="Paramètres de l'algorithme : ", style={'text-decoration': 'underline'}),
        html.P('Nombre de splits :'),
        dcc.Input(id='nb_splits', value=5, type='number', min=1, max=20, step=1),
        html.Br(),
        html.Br(),
        html.P('Nombre de répétitions :'),
        dcc.Input(id='nb_repeats', value=20, type='number', min=1, max=250, step=1),
        html.Br(),
        html.Br(),
    ], style={'margin-left': '30px', 'margin-top': '30px', 'textAlign':'center'}),
    html.Div(html.Button("Lancer l'algorithme", id='submit-regmul', n_clicks=0, className="buttonClick"), style={'textAlign': 'center', 'display': 'block'}),
    html.Div(id='analyse_regmul'),
])
    
# ElasticNet
elasticnet_layout = html.Div(children=[html.Div(children=
    [
        html.Div(children=[
            html.H6(children="Paramètres validation croisée : ", style={'text-decoration': 'underline'}),
            html.P('Nombre de splits :'),
            dcc.Input(id='nb_splits', value=5, type='number', min=1, max=20, step=1),
            html.Br(),
            html.Br(),
            html.P('Nombre de répétitions :'),
            dcc.Input(id='nb_repeats', value=20, type='number', min=1, max=250, step=1),
            html.Br(),
            html.Br(),
        ]),
        html.Div(children=[
            html.H6(children="Paramètres de l'algorithme : ", style={'text-decoration': 'underline'}),
            html.P("Nombre d'itérations :"),
            dcc.Input(id='iterations', value=100, type='number', min=1, max=1000, step=1), 
            html.Br(),
            html.Br(),
            html.P("Hyper-paramètre Alpha :"),
            dcc.Input(id='alpha', value=0.1, type='number', min=0, max=20, step=0.01), 
            html.Br(),
            html.Br(),
            html.P("L1 ratio - curseur de répartition des pénalités Ridge et Lasso : "),
            html.P("1 = Lasso, 0 = Ridge. Entre les 2, combinaisons des 2."),
            dcc.Input(id='l1_ratio', value=0.5, type='number', min=0, max=1, step=0.05),
            html.Br(),
            html.Br(),

        ], style={'margin-left': '50px'}),
    ], style ={'margin-left':'150px', 'margin-top': '30px', 'display':'flex', 'justify-content':'center'}),
    html.Div(html.Button("Lancer l'algorithme", id='submit-elastic', n_clicks=0, className="buttonClick"), style={'textAlign': 'center', 'display': 'block'}),
    html.Div(id='analyse_elastinet')])

# KNN
knn_layout = html.Div(children=[html.Div(children=
    [
        html.H6(children="Paramètres de l'algorithme : ", style={'text-decoration': 'underline'}),
        html.P('Nombre de splits :'),
        dcc.Input(id='nb_splits', value=5, type='number', min=1, max=20, step=1),
        html.Br(),
        html.Br(),
        html.P('Nombre de répétitions :'),
        dcc.Input(id='nb_repeats', value=20, type='number', min=1, max=250, step=1),
        html.Br(),
        html.Br(),
        html.P('Nombre de voisins :'),
        dcc.Input(id='K', value=7, type='number', min=1, max=50, step=1),
        html.Br(),
        html.Br(),

    ], style={'margin-left': '30px', 'margin-top': '30px', 'textAlign':'center'}),
    html.Div(html.Button("Lancer l'algorithme", id='submit-knn', n_clicks=0, className="buttonClick"), style={'textAlign': 'center', 'display': 'block'}),
    html.Div(id='analyse_knn'),
])