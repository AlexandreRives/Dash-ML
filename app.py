import base64
import datetime
import io
import dash
import os
from dash.html.Br import Br
import pandas as pd
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
from dash import dash_table
from pandas.io.formats import style


# CSS
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Running dash
app = dash.Dash(__name__, external_stylesheets=[external_stylesheets, dbc.themes.BOOTSTRAP])

# Layout
app.layout = html.Div([
    html.Div([html.H2(children='Bienvenue sur notre application', style={'textAlign': 'center'}),
    html.H5(children="Elisa, Jackie, Alexandre", style={'textAlign': 'center'}),
    html.Br()
    ]),
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
        # Only one file can be uploaded
        multiple=True
    ),
    
    html.Div(id='output-data-upload'),
    html.Hr(),
])

# Fonction qui permet le chargement et l'affichage du dataset + des options de sélections de variables
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
        html.P("Nom du fichier : " + filename),

        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            page_size=5
        ),

        html.Hr(),  # horizontal line

                html.H6(children="Veuillez choisir votre variable cible : ", style={'text-decoration': 'underline'}),
        dcc.Dropdown(
            id='varY',
            options=[
                {'label': i, 'value': i} for i in df.columns
            ],
            style={'width': '300px'}
        ),
        html.Hr(),
        html.H6(children="Veuillez choisir vos variables explicatives : ", style={'text-decoration': 'underline'}),
        dcc.Dropdown(
            id='varX',
            options=[
                {'label': i, 'value': i} for i in df.columns
            ],
            multi=True,
            style={'width': '300px'}
        ),
        html.Hr(),

        html.H6(children="Veuillez choisir votre algorithme : ", style={'text-decoration': 'underline'}),

        dcc.Tabs(id="algos", value='liste_algos', children=[
            dcc.Tab(label='KMeans', value='KMeans'),
            dcc.Tab(label='Arbre de décision', value='arbre'),
            dcc.Tab(label='CAH', value='cah'),
            dcc.Tab(label='Analyse Discriminante', value='adl'),
            dcc.Tab(label='Régression logisitque', value='reglog'),
            dcc.Tab(label='Régression Linéaire Multiple', value='regmul'),
        ]),

        html.Hr(),
    ])

#Call for the importing file
@app.callback(  Output('output-data-upload', 'children'),
                Input('upload-data', 'contents'),
                State('upload-data', 'filename'),
                State('upload-data', 'last_modified'))

#Update datatable when changing the doc.
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

def update_varX():
    return 0

#Run server
if __name__ == '__main__':
    app.run_server(debug=True)