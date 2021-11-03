import base64
import io
import dash
import pandas as pd
import dash_bootstrap_components as dbc
import classification_layout
import regression_layout

from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
from dash import dash_table
from classification import Classification
from regression import Regression

# CSS
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Instanciation de Dash
app = dash.Dash(__name__, external_stylesheets=[external_stylesheets, dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Cadre pour le chargement du fichier
app_layout = html.Div(children=
    [
        dcc.Upload(
        id='upload-data',
        children=html.Div(
            [
                '1. Faites glisser votre fichier ici ou ',
                html.A('cliquer pour choisir votre fichier')
            ]),
        style={
            'width': 'auto',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin-top': '10px',
            'margin-bottom' : '10px',
            'margin-left': '200px',
            'margin-right': '200px',
            'textAlign': 'center',
            'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414',
            'color':'#333'
        },
        # Seulement un fichier peut être sélectionné.
        multiple=True
    ),

    html.Div(children=
        [
            html.A('Appuyer ici avant de charger un nouveau dataset', href='/', style={'textAlign': 'center', 'display': 'block', 'margin-top':'30px'}, className="app-header--refresh"),
            html.Br(),
        ]
    ),

    # Affichage du tableau
    html.Div(id='output-data-upload'),
    html.Hr(),

    # Onglets de sélection des algorithmes
    dcc.Tabs(id="algos", children=
        [
            dcc.Tab(label='Support Vecteur Machine', value='svm', style={'color':'black', 'text-shadow': '0px 0px 5px red', 'font-size':'1.1em'}),
            dcc.Tab(label='Analyse Discriminante Linéaire', value='adl', style={'color':'black', 'text-shadow': '0px 0px 5px red', 'font-size':'1.1em'}),
            dcc.Tab(label='Régression Logisitque', value='reglog', style={'color':'black', 'text-shadow': '0px 0px 5px red', 'font-size':'1.1em'}),
            dcc.Tab(label='K Plus Proches Voisins', value='knn', style={'color':'black', 'text-shadow': '0px 0px 5px blue', 'font-size':'1.1em'}),
            dcc.Tab(label='Arbre de Décision', value='arbre', style={'color':'black', 'text-shadow': '0px 0px 5px blue', 'font-size':'1.1em'}),
            dcc.Tab(label='Régression Linéaire Multiple', value='regmul', style={'color':'black', 'text-shadow': '0px 0px 5px blue', 'font-size':'1.1em'}),
        ]
    ),

    html.Div(id='contenu_algo'),
    html.Hr(),
    
    ]
)

# Layout
app.layout = html.Div(children=
    [
        html.Br(),
        html.H2('Bienvenue sur notre application', style={'textAlign': 'center', 'margin-top': '20px', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
        html.H3("Elisa, Jacky, Alexandre", style={'textAlign': 'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
        html.Br(),
        html.Div(app_layout)
    ]
)

# Fonction qui permet le chargement et l'affichage du dataset + des options de sélection de variables.
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # File : iris.csv
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), sep="[,;]", engine='python')
        elif 'xls' in filename:
            # Excel file :
            df = pd.read_excel(io.BytesIO(decoded), sep=";")
    except Exception as e:
        print(e)
        return html.Div([
            'Erreur lors du chargement de votre fichier.'
        ])

    return html.Div(children=
        [
            html.Div(dash_table.DataTable(
                id='df',
                data=df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df.columns],
                page_size=5,
                style_header={
                    'backgroundColor': 'rgb(30, 30, 30)',
                    'color': 'white',
                },
                style_cell={'textAlign':'center', 
                            'overflow': 'hidden',
                            'textOverflow': 'ellipsis',
                            'maxWidth': 0},
                style_data={'backgroundColor': 'rgb(50, 50, 50)',
                            'color': 'white',
                            'width': '20px'
                },
            ), style={'margin': '20px'}),

            html.Hr(),
            html.Br(),
            html.P("2. Choix des variables", style={'textAlign': 'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
            
            html.Div(children=
                [
                    html.H6(children="Choisir votre variable cible : ", style={'text-decoration': 'underline', 'margin-left': '10px'}),
                    dcc.Dropdown(
                        id='varY',
                        options=
                        [
                            {'label': i, 'value': i} for i in df.columns
                        ],
                        style={'width': '300px', 'margin-left': '10px'},
                    ),
                    html.Br(),
                    html.H6(children="Choisir vos variables explicatives : ", style={'text-decoration': 'underline', 'margin-left':'10px'}),
                    dcc.Dropdown(
                        id='varX',
                        options=
                        [
                            {'label': i, 'value': i} for i in df.columns
                        ],
                        multi=True,
                        style={'width': '300px', 'margin-left': '10px'}
                    ),
                    html.Br(),
                ]
            ),
            html.Br(),

            html.Div(html.P('***** Les onglets de couleurs rouges correspondent aux algorithmes de types classification tandis que les bleus correspondent aux régressions *****'), style={'textAlign':'center', 'fontWeight':'bold'}),

            html.Br(),

            html.P("3. Choix de l'algorithme", style={'textAlign': 'center', 'text-shadow':'-1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff, 1px 1px 10px #141414', 'color':'#333'}),
        ]
    )

# Mis à jour du tableau en fonction du fichier qui est importé
@app.callback(Output('output-data-upload', 'children'),
                Input('upload-data', 'contents'),
                State('upload-data', 'filename'),
                State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children=[
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

# Fonction qui implémente les onglets et les options de chaque algorithme
@app.callback(Output('contenu_algo', 'children'), 
                Input('algos', 'value'))
def render_algo(onglets):

    # Support Vector Machine
    if onglets == 'svm':
        return classification_layout.SVM_layout

    # Régression logistique
    elif onglets == 'reglog':
        return classification_layout.reglog_layout

    # Analyse disciminante linéaire
    elif onglets == 'adl':
        # Mise en forme : séparer en plusieurs children. Voir : https://dash.plotly.com/layout
        return classification_layout.adl_layout

    # K plus proches voisins
    elif onglets == 'knn':
        return regression_layout.knn_layout
   
        # Arbre de décision
    elif onglets == 'arbre':
        return regression_layout.arbre_layout

    # Régression linéraire multiple
    elif onglets == 'regmul':
        return regression_layout.regmul_layout

# Bouton submit analyse avec SVM
@app.callback(Output('analyse_svm', 'children'),
                State('varY', 'value'),
                State('varX', 'value'),
                State('df', 'data'),
                Input('submit-svm', 'n_clicks'))
def affichage_algo_svm(varY, varX, df, clusters, n_clicks):
    if(n_clicks != 0):

        return html.Br(),html.Div(children=[html.H5("Présentation de l'algorithme du support vecteur machine", style={'textAlign': 'center'})]),

# Bouton submit avec ADL
@app.callback(Output('analyse_adl', 'children'),
                State('varY', 'value'),
                State('varX', 'value'),
                State('df', 'data'),
                State('nb_splits', 'value'),
                State('t_ech_test', 'value'),
                State('solv', 'value'),
                State('nb_repeats', 'value'),
                Input('submit-adl', 'n_clicks'))
def affichage_algo_adl(varY, varX, df, nb_splits, t_ech_test, solv, nb_repeats, n_clicks):
    if(n_clicks != 0):
        df = pd.DataFrame(df)

        return html.Br(),html.Div(children=[html.H5("Présentation de l'algorithme de l'analyse discriminante linéaire", style={'textAlign': 'center'})]),

# Bouton submit avec Reg Log
@app.callback(Output('analyse_reglog', 'children'),
                State('varY', 'value'),
                State('varX', 'value'),
                State('df', 'data'),
                Input('submit-reglog', 'n_clicks'))
def affichage_algo_reglog(varY, varX, df, clusters, n_clicks):
    if(n_clicks != 0):
        df = pd.DataFrame(df)
        
        return html.Br(),html.Div(children=[html.H5("Présentation de l'algorithme de la régression logistique", style={'textAlign': 'center'})]),

# Bouton submit avec KNN
@app.callback(Output('analyse_knn', 'children'),
                State('varY', 'value'),
                State('varX', 'value'),
                State('df', 'data'),
                Input('submit-knn', 'n_clicks'))
def affichage_algo_knn(varY, varX, df, clusters, n_clicks):
    if(n_clicks != 0):
        df = pd.DataFrame(df)

        return html.Br(),html.Div(children=[html.H5("Présentation de l'algorithme de la classification ascendante hiérarchique", style={'textAlign': 'center'})]),

# Bouton submit analyse avec Arbre des décision
@app.callback(Output('analyse_arbre', 'children'),
                State('varY', 'value'),
                State('varX', 'value'),
                State('df', 'data'),
                State('nb_feuilles', 'value'),
                State('nb_individus', 'value'),
                State('nb_splits', 'value'),
                State('nb_repeats', 'value'),
                Input('submit-arbre', 'n_clicks'))
def affichage_algo_arbre(varY, varX, df, nb_feuilles, nb_individus, nb_splits, nb_repeats, n_clicks):
    if(n_clicks != 0):
        df = pd.DataFrame(df)
        arbre = Regression(df, varX, varY)
        return arbre.algo_arbre(df, varX, varY, nb_feuilles, nb_individus, nb_splits, nb_repeats)

# Bouton submit avec Reg Mul
@app.callback(Output('analyse_regmul', 'children'),
                State('varY', 'value'),
                State('varX', 'value'),
                State('df', 'data'),
                State('nb_variables', 'value'),
                State('nb_splits', 'value'),
                State('nb_repeats', 'value'),
                Input('submit-regmul', 'n_clicks'))
def affichage_algo_regmul(varY, varX, df, nb_variables, nb_splits, nb_repeats, n_clicks):
    if(n_clicks != 0):
        df = pd.DataFrame(df)
        algo_reg_mul = Regression(df, varX, varY)
        return algo_reg_mul.regression_lineaire_multiple(df, varX, varY, nb_variables, nb_splits, nb_repeats)

# Lancement du serveur
if __name__ == '__main__':
    app.run_server(debug=True)