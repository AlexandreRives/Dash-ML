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

# CSS
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Lancement de dash
app = dash.Dash(__name__, external_stylesheets=[external_stylesheets, dbc.themes.BOOTSTRAP])

multi_select = []
select = ''

# Layout
app.layout = html.Div([
    html.Div(children=[html.H2('Bienvenue sur notre application', style={'textAlign': 'center'}),
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
            'margin': '10px'
        },
        # Seulement un fichier peut être sélectionné.
        multiple=True
    ),
    
    html.Div(id='output-data-upload'),
    html.Hr(),

    # Onglets de sélections des algorithmes
    dcc.Tabs(id="algos", value='KMeans', children=[
        dcc.Tab(label='KMeans', value='KMeans'),
        dcc.Tab(label='Arbre de décision', value='arbre'),
        dcc.Tab(label='CAH', value='cah'),
        dcc.Tab(label='Analyse Discriminante', value='adl'),
        dcc.Tab(label='Régression logisitque', value='reglog'),
        dcc.Tab(label='Régression Linéaire Multiple', value='regmul'),
    ]),
    html.Div(id='contenu_algo'),

    html.Hr(),
])


# Fonction qui permet le chargement et l'affichage du dataset + des options de sélection de variables.
#@app.callback()
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

    return html.Div([
        html.P("Nom du fichier : " + filename, style={'margin-left': '10px'}),

        dash_table.DataTable(
            id='df',
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            page_size=5
        ),

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
@app.callback(Output('contenu_algo', 'children'), Input('algos', 'value'))
def render_algo(onglets):
    # Récupérer les Y
    # Récupérer les X
    if onglets == 'KMeans':
        return html.Div(children=[
            html.P('Choisir le nombre de cluster :'),
            dcc.Input(id='range', type='number', min=1, max=10, step=1),
            html.Br(),
            html.Br(),
            html.Button("Lancer l'algorithme", id='submit-val', n_clicks=0)
        ], style={'margin-left': '10px'})
    elif onglets == 'arbre':
        return html.Div(children=[
            html.Br(),
            html.Br(),
            html.Button("Lancer l'algorithme", id='submit-val', n_clicks=0)
        ], style={'margin-left': '10px'})
    elif onglets == 'cah':
        return html.Div(children=[
            html.Br(),
            html.Br(),
            html.Button("Lancer l'algorithme", id='submit-val', n_clicks=0)
        ], style={'margin-left': '10px'})
    elif onglets == 'adl':
        return html.Div(children=[
            html.Br(),
            html.Br(),
            html.Button("Lancer l'algorithme", id='submit-val', n_clicks=0)
        ], style={'margin-left': '10px'})
    elif onglets == 'reglog':
        return html.Div(children=[
            html.Br(),
            html.Br(),
            html.Button("Lancer l'algorithme", id='submit-val', n_clicks=0)
        ], style={'margin-left': '10px'})
    else:
        return html.Div(children=[
            html.Br(),
            html.Br(),
            html.Button("Lancer l'algorithme", id='submit-val', n_clicks=0)
        ], style={'margin-left': '10px'})

#Lancement du serveur
if __name__ == '__main__':
    app.run_server(debug=True)