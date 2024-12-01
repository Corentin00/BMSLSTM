import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objs as go
from flask import Flask

# Création du serveur Flask
server = Flask(__name__)

# Initialisation de l'application Dash
app = dash.Dash(__name__, server=server, url_base_pathname='/')

# Génération de données de démonstration
def generate_demo_data():
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='M')
    sales = np.cumsum(np.random.normal(1000, 500, len(dates)))
    categories = np.random.choice(['Produit A', 'Produit B', 'Produit C'], len(dates))
    
    df = pd.DataFrame({
        'Date': dates,
        'Sales': sales,
        'Category': categories
    })
    return df

# Charger les données
df = generate_demo_data()

# Mise en page de l'application
app.layout = html.Div([
    html.H1('Tableau de Bord de Visualisation de Ventes', 
            style={'textAlign': 'center', 'color': '#503D36', 'fontSize': 40}),
    
    # Section de filtres
    html.Div([
        html.Label('Sélectionner une catégorie :'),
        dcc.Dropdown(
            id='category-dropdown',
            options=[{'label': cat, 'value': cat} for cat in df['Category'].unique()],
            value=df['Category'].unique()[0],
            style={'width': '50%'}
        )
    ], style={'margin': '20px'}),
    
    # Conteneur pour les graphiques
    html.Div([
        # Graphique de série temporelle
        html.Div([
            dcc.Graph(id='time-series-chart')
        ], className='six columns'),
        
        # Graphique de distribution
        html.Div([
            dcc.Graph(id='sales-distribution')
        ], className='six columns')
    ], className='row'),
    
    # Tableau de données
    html.Div([
        dash_table.DataTable(
            id='data-table',
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'backgroundColor': 'rgb(250, 250, 250)'
            },
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            }
        )
    ], style={'margin': '20px'})
])

# Callbacks pour mettre à jour les graphiques
@app.callback(
    [Output('time-series-chart', 'figure'),
     Output('sales-distribution', 'figure')],
    [Input('category-dropdown', 'value')]
)
def update_graphs(selected_category):
    # Filtrer les données par catégorie
    filtered_df = df[df['Category'] == selected_category]
    
    # Graphique de série temporelle
    time_series_fig = px.line(
        filtered_df, 
        x='Date', 
        y='Sales', 
        title=f'Évolution des Ventes - {selected_category}',
        labels={'Sales': 'Montant des Ventes'}
    )
    time_series_fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Graphique de distribution
    distribution_fig = px.box(
        df, 
        x='Category', 
        y='Sales', 
        title='Distribution des Ventes par Catégorie'
    )
    distribution_fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return time_series_fig, distribution_fig

# Point d'entrée pour exécuter l'application
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)