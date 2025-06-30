import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import numpy as np

app = dash.Dash(__name__)
server = app.server  # utile per deploy futuri

# Genera una matrice 10x10 casuale
def generate_matrix():
    return np.random.rand(10, 10)

# Inizializza la matrice
initial_matrix = generate_matrix()

app.layout = html.Div([
    html.H2("Zappa le righe cliccando!"),
    dcc.Graph(id='heatmap'),
    html.Button('Reset', id='reset-btn', n_clicks=0),
    dcc.Store(id='matrix-store', data=initial_matrix.tolist()),
    dcc.Store(id='zapped-rows', data=[]),
])

@app.callback(
    Output('heatmap', 'figure'),
    Input('matrix-store', 'data'),
)
def update_heatmap(matrix_data):
    matrix = np.array(matrix_data)
    fig = go.Figure(data=go.Heatmap(z=matrix, colorscale='Viridis'))
    fig.update_layout(clickmode='event+select', height=500)
    return fig

@app.callback(
    Output('matrix-store', 'data'),
    Output('zapped-rows', 'data'),
    Input('heatmap', 'clickData'),
    Input('reset-btn', 'n_clicks'),
    State('matrix-store', 'data'),
    State('zapped-rows', 'data'),
    prevent_initial_call=True
)
def zap_row(click_data, reset_clicks, matrix_data, zapped_rows):
    ctx = dash.callback_context
    matrix = np.array(matrix_data)

    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'reset-btn':
        return generate_matrix().tolist(), []
    
    if triggered_id == 'heatmap' and click_data:
        y = click_data['points'][0]['y']  # y is the row index
        if y not in zapped_rows:
            matrix[y, :] = np.nan
            zapped_rows.append(y)
    
    return matrix.tolist(), zapped_rows

if __name__ == '__main__':
    app.run(debug=True)
