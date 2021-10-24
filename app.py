import base64
import io
import dash
import pandas as pd
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
from dash import dash_table
import warnings
warnings.filterwarnings("ignore")
from kmeans import KMeans_algo

# CSS
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Lancement de dash
app = dash.Dash(__name__, external_stylesheets=[external_stylesheets, dbc.themes.BOOTSTRAP])

# Layout
app.layout = html.Div([
    html.Div(children=[html.H2('Bienvenue sur notre application', style={'textAlign': 'center', 'margin-top': '20px'}),
    html.H5("Elisa, Jackie, Alexandre", style={'textAlign': 'center'}),
    html.Br()
    ]),
    
    # Cadre pour le chargement du fichier
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Faites glisser votre fichier ici ou ',
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
            'margin-right': '200px'
        },
        # Seulement un fichier peut être sélectionné.
        multiple=True
    ),
    
    html.A('Appuyer ici avant de charger un nouveau dataset', href='/', style={'textAlign': 'center', 'display': 'block', 'margin-top':'30px'}, className="app-header--refresh"),   

    # Affichage du tableau
    html.Div(id='output-data-upload'),
    html.Hr(),

    # Onglets de sélection des algorithmes
    dcc.Tabs(id="algos", children=[
        dcc.Tab(label='KMeans', value='KMeans'),
        dcc.Tab(label='Arbre de décision', value='arbre'),
        dcc.Tab(label='CAH', value='cah'),
        dcc.Tab(label='Analyse Discriminante', value='adl'),
        dcc.Tab(label='Régression logisitque', value='reglog'),
        dcc.Tab(label='Régression Linéaire Multiple', value='regmul'),
    ]),
    html.Div(id='contenu_algo'),

    html.Hr(),

    html.Div(id='analyse_kmeans'),
    html.Div(id='analyse_arbre'),
    html.Div(id='analyse_cah'),
    html.Div(id='analyse_adl'),
    html.Div(id='analyse_reglog'),
    html.Div(id='analyse_regmul'),

])

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

    return html.Div(children=[
        html.P("Nom du fichier : " + filename, style={'margin-left': '10px'}),

        html.Div(dash_table.DataTable(
            id='df',
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            page_size=5
        ), style={'margin':'20px'}),

        html.Hr(),  # horizontal line

        html.H6(children="Choisir votre variable cible : ", style={'text-decoration': 'underline', 'margin-left': '10px'}),
        dcc.Dropdown(
            id='varY',
            options=[
                {'label': i, 'value': i} for i in df.columns
            ],
            style={'width': '300px', 'margin-left': '10px'}
        ),
        html.Hr(),
        html.H6(children="Choisir vos variables explicatives : ", style={'text-decoration': 'underline', 'margin-left': '10px'}),
        dcc.Dropdown(
            id='varX',
            options=[
                {'label': i, 'value': i} for i in df.columns
            ],
            multi=True,
            style={'width': '300px', 'margin-left': '10px'}
        ),
        html.Hr(),
    ])

#Mis à jour du tableau en fonction du fichier qui est importé
@app.callback(Output('output-data-upload', 'children'),
                Input('upload-data', 'contents'),
                State('upload-data', 'filename'),
                State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

#Fonction qui implémente les onglets et les options de chaque algorithme
@app.callback(Output('contenu_algo', 'children'), 
                Input('algos', 'value'))
def render_algo(onglets):

    #KMeans
    if onglets == 'KMeans':
        return html.Div(children=[
            html.P('Choisir le nombre de clusters :'),
            dcc.Input(id='cluster', value='nb_clusters', type='number', min=1, max=10, step=1),
            html.Br(),
            html.Br(),
            html.Div(html.Button("Lancer l'algorithme", id='submit-kmeans', n_clicks=0, className="buttonClick"), style={'textAlign': 'center', 'display': 'block'})
        ], style={'margin-left': '10px', 'margin-top': '30px'})

    # Arbre de décision
    elif onglets == 'arbre':
        return html.Div(children=[
            html.P('Choisir le nombre de feuilles :'),
            dcc.Input(id='feuilles', value='nb_feuilles', type='number', min=1, max=10, step=1),
            html.Br(),
            html.Br(),
            html.Div(html.Button("Lancer l'algorithme", id='submit-arbre', n_clicks=0, className="buttonClick"), style={'textAlign': 'center', 'display': 'block'})
        ], style={'margin-left': '10px', 'margin-top': '30px'})

    # Classification ascendante hiérarchique
    elif onglets == 'cah':
        return html.Div(children=[
            html.Br(),
            html.Br(),
            html.Div(html.Button("Lancer l'algorithme", id='submit-cah', n_clicks=0, className="buttonClick"), style={'textAlign': 'center', 'display': 'block'})
        ], style={'margin-left': '10px', 'margin-top': '30px'})

    # Analyse disciminante linéaire
    elif onglets == 'adl':
        return html.Div(children=[
            html.Br(),
            html.Br(),
            html.Div(html.Button("Lancer l'algorithme", id='submit-adl', n_clicks=0, className="buttonClick"), style={'textAlign': 'center', 'display': 'block'})
        ], style={'margin-left': '10px', 'margin-top': '30px'})

    # Régression logistique
    elif onglets == 'reglog':
        return html.Div(children=[
            html.Br(),
            html.Br(),
            html.Div(html.Button("Lancer l'algorithme", id='submit-reglog', n_clicks=0, className="buttonClick"), style={'textAlign': 'center', 'display': 'block'})
        ], style={'margin-left': '10px', 'margin-top': '30px'})

    # Régression linéraire multiple
    elif onglets == 'regmul':
        return html.Div(children=[
            html.Br(),
            html.Br(),
            html.Div(html.Button("Lancer l'algorithme", id='submit-regmul', n_clicks=0, className="buttonClick"), style={'textAlign': 'center', 'display': 'block'})
        ], style={'margin-left': '10px', 'margin-top': '30px'})

# Bouton submit analyse KMeans
@app.callback(Output('analyse_kmeans', 'children'),
                Input('varY', 'value'),
                Input('varX', 'value'),
                Input('df', 'data'),
                Input('cluster', 'value'),
                Input('submit-kmeans', 'n_clicks'))
def affichage_algo_kmeans(varY, varX, df, clusters, n_clicks):
    if(n_clicks != 0):
        df = pd.DataFrame(df)
        #Retraiter les données pour les envoyer dans l'algo.
        algoKmeans = KMeans_algo(df, varX, varY, clusters)
        algoKmeans.Algo_KMeans(df, varX, varY, clusters)
        return html.Div(children=[html.H5("Présentation de l'algorithme des KMeans", style={'textAlign': 'center'})]),

# Bouton submit analyse avec Arbre des décisions
@app.callback(Output('analyse_arbre', 'children'),
                Input('varY', 'value'),
                Input('varX', 'value'),
                Input('df', 'data'),
                Input('feuilles', 'value'),
                Input('submit-arbre', 'n_clicks'))
def affichage_algo_arbre(varY, varX, df, clusters, n_clicks):
    if(n_clicks != 0):
        df = pd.DataFrame(df)

        return html.Div(children=[html.H5("Présentation de l'algorithme de l'arbre des décisions", style={'textAlign': 'center'})]),

#Bouton submit avec CAH
@app.callback(Output('analyse_cah', 'children'),
                Input('varY', 'value'),
                Input('varX', 'value'),
                Input('df', 'data'),
                Input('submit-cah', 'n_clicks'))
def affichage_algo_cah(varY, varX, df, clusters, n_clicks):
    if(n_clicks != 0):
        df = pd.DataFrame(df)

        return html.Div(children=[html.H5("Présentation de l'algorithme de la classification ascendante hiérarchique", style={'textAlign': 'center'})]),

#Bouton submit avec ADL
@app.callback(Output('analyse_adl', 'children'),
                Input('varY', 'value'),
                Input('varX', 'value'),
                Input('df', 'data'),
                Input('submit-adl', 'n_clicks'))
def affichage_algo_adl(varY, varX, df, clusters, n_clicks):
    if(n_clicks != 0):
        df = pd.DataFrame(df)
        #Retraiter les données pour les envoyer dans l'algo.

        return html.Div(children=[html.H5("Présentation de l'algorithme de l'analyse discriminante linéaire", style={'textAlign': 'center'})]),

#Bouton submit avec Reg Log
@app.callback(Output('analyse_reglog', 'children'),
                Input('varY', 'value'),
                Input('varX', 'value'),
                Input('df', 'data'),
                Input('submit-reglog', 'n_clicks'))
def affichage_algo_reglog(varY, varX, df, clusters, n_clicks):
    if(n_clicks != 0):
        df = pd.DataFrame(df)
        #Retraiter les données pour les envoyer dans l'algo.
        # algoKmeans = KMeans_algo(df, varX, varY, clusters)
        # algoKmeans.Algo_KMeans(df, varX, varY, clusters)
        return html.Div(children=[html.H5("Présentation de l'algorithme de la régression logistique", style={'textAlign': 'center'})]),

#Bouton submit avec Reg Mul
@app.callback(Output('analyse_regmul', 'children'),
                Input('varY', 'value'),
                Input('varX', 'value'),
                Input('df', 'data'),
                Input('submit-regmul', 'n_clicks'))
def affichage_algo_regmul(varY, varX, df, clusters, n_clicks):
    if(n_clicks != 0):
        df = pd.DataFrame(df)
        #Retraiter les données pour les envoyer dans l'algo.
        # algoKmeans = KMeans_algo(df, varX, varY, clusters)
        # algoKmeans.Algo_KMeans(df, varX, varY, clusters)
        return html.Div(children=[html.H5("Présentation de l'algorithme de la regression linéaire multiple", style={'textAlign': 'center'})]),



#Lancement du serveur
if __name__ == '__main__':
    app.run_server(debug=True)