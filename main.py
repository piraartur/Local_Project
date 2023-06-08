import pandas as pd
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import plotly.graph_objects as go
from dash.dependencies import State

deepfaceCSV = pd.read_csv('models/deepface/csv_files/emotions_deepface.csv')
ferCSV = pd.read_csv('models/fer/csv_files/emotions_fer.csv')

app = Dash(__name__, assets_folder='assets')
app.layout = html.Div([
    html.Div(children='Facial Expressions Results', className='title'),
    html.Hr(),
    dcc.RadioItems(
        options=[{'label': key, 'value': key} for key in deepfaceCSV.columns],
        value='',
        id='controls-and-radio-item',
        className='radio-buttons'
    ),
    html.Button("Show All", id="show-all-btn", n_clicks=0, className="show-all-btn"),
    dcc.Graph(figure={}, id='deepfaceModel', className='deepfaceModel'),
    dcc.Graph(figure={}, id='ferModel', className='ferModel')
])

app.config['suppress_callback_exceptions'] = True

@app.callback(
    [
        Output('deepfaceModel', 'figure'),
        Output('ferModel', 'figure'),
        Output('show-all-btn', 'n_clicks'),
        Output('controls-and-radio-item', 'value')
    ],
    [
        Input('controls-and-radio-item', 'value'),
        Input('show-all-btn', 'n_clicks')
    ],
    [State('controls-and-radio-item', 'value')]
)


def update_graph(emotion_chosen, show_all_clicks, current_value):

    if show_all_clicks is None:
        show_all_clicks = 0

    if show_all_clicks > 0:
        emotion_chosen = ''
        show_all_clicks -= 1
        current_value = ''

    if current_value:
        emotion_chosen = current_value

    fig = go.Figure()
    fig2 = go.Figure()

    if emotion_chosen == '':
        for col in deepfaceCSV.columns:
            fig.add_trace(go.Bar(
                x=[col],
                y=[deepfaceCSV.loc[0][col]],
                marker=dict(color="#28fc03"),
            ))
            fig2.add_trace(go.Bar(
                x=[col],
                y=[ferCSV.loc[0][col]],
                marker=dict(color="royalblue"),
            ))

    else:
        for col in deepfaceCSV.columns:
            if col == emotion_chosen:
                fig.add_trace(go.Bar(
                    x=[col],
                    y=[deepfaceCSV.loc[0][col]],
                    marker=dict(color="#28fc03"),
                ))
                fig2.add_trace(go.Bar(
                    x=[col],
                    y=[ferCSV.loc[0][col]],
                    marker=dict(color="royalblue"),
                ))
            else:
                fig.add_trace(go.Bar(
                    x=[col],
                    y=[0],
                    marker=dict(color='#28fc03'),
                ))
                fig2.add_trace(go.Bar(
                    x=[col],
                    y=[0],
                    marker=dict(color='royalblue'),
                ))



    fig.update_layout(
        title='Deepface Model',
        xaxis=dict(gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray'),
        plot_bgcolor='#333',
        paper_bgcolor='#333',
        title_font=dict(color='white', size=20),
        title_x=0.5,
    )
    fig2.update_layout(
        title='Fer Model',
        xaxis=dict(gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray'),
        plot_bgcolor='#333',
        paper_bgcolor='#333',
        title_font=dict(color='white', size=20),
        title_x=0.5
    )
    fig2.update_traces(marker=dict(color="royalblue"))

    fig.update_xaxes(title='Emotion', color='white')
    fig.update_yaxes(title='Percentage', color='white')
    fig2.update_xaxes(title='Emotion', color='white')
    fig2.update_yaxes(title='Percentage', color='white')
    return fig, fig2, show_all_clicks, emotion_chosen



if __name__ == '__main__':
    app.run_server(debug=True)