import dash
import dash_html_components as html

cah_layout = html.Div(children=[
            html.Br(),
            html.Br(),
            html.Div(html.Button("Lancer l'algorithme", id='submit-cah', n_clicks=0, className="buttonClick"), style={'textAlign': 'center', 'display': 'block'}),
            html.Div(id='analyse_cah'),
        ], style={'margin-left': '10px', 'margin-top': '30px'})