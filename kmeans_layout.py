import dash
import dash_html_components as html
import dash_core_components as dcc

kmeans_layout = html.Div(children=[
            html.P('Choisir le nombre de clusters :'),
            dcc.Input(id='nb_cluster', value=1, type='number', min=1, max=10, step=1),
            html.Br(),
            html.Br(),
            html.Div(html.Button("Lancer l'algorithme", id='submit-kmeans', n_clicks=0, className="buttonClick"), style={'textAlign': 'center', 'display': 'block'}),
            html.Div(id='analyse_kmeans'),
        ], style={'margin-left': '10px', 'margin-top': '30px'})


