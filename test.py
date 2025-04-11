import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import numpy as np

# Initialize the app
app = dash.Dash(__name__)
app.title = "Zappable Heatmap"

# Global array
Z = np.random.rand(10, 10)
zapped_rows = []

def get_figure():
    fig = go.Figure(data=go.Heatmap(z=Z, colorscale='Viridis'))
    fig.update_layout(
        title="Click a row to zap it",
        clickmode='event+select',
        xaxis_title="Column",
        yaxis_title="Row",
        yaxis=dict(autorange='reversed')
    )
    return fig

app.layout = html.Div([
    html.H1("Zappable Heatmap"),
    dcc.Graph(id='heatmap', figure=get_figure(), config={"displayModeBar": False}),
    html.Div(id='clicked-output'),
    html.Button("Reset", id='reset-btn', n_clicks=0)
])

@app.callback(
    Output('heatmap', 'figure'),
    Output('clicked-output', 'children'),
    Input('heatmap', 'clickData'),
    Input('reset-btn', 'n_clicks'),
    prevent_initial_call=True
)
def update_heatmap(clickData, n_clicks):
    global Z, zapped_rows
    ctx = dash.callback_context

    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'reset-btn':
        Z = np.random.rand(10, 10)
        zapped_rows = []
        return get_figure(), "Heatmap reset."

    if clickData and 'points' in clickData:
        point = clickData['points'][0]
        row = point['y']
        if row not in zapped_rows:
            Z[row, :] = np.nan
            zapped_rows.append(row)
        return get_figure(), f"Zapped row {row}"

    return dash.no_update, dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True)
