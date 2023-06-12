import pandas as pd
from dash import Dash, html, dcc, Output, Input
import plotly.graph_objects as go

deepfaceCSV = pd.read_csv('models/deepface/csv_files/emotions_deepface.csv')
ferCSV = pd.read_csv('models/fer/csv_files/emotions_fer.csv')
cnnCSV = pd.read_csv('models/CNN/csv_files/emotion_results.csv')

column_names = deepfaceCSV.columns.tolist()
options = [{'label': key, 'value': key} for key in column_names]

app = Dash(__name__, assets_folder='assets')
app.layout = html.Div([
    html.Div(children='Facial Expressions Results', className='title'),
    html.Hr(),
    html.Div(className='sticky-div', children=[
        html.Button("Show All", id="show-all-btn", n_clicks=0, className="show-all-btn")
    ]),
    dcc.Graph(figure={}, id='deepfaceModel', className='deepfaceModel'),
    dcc.Graph(figure={}, id='ferModel', className='ferModel'),
    dcc.Graph(figure={}, id='cnnModel', className='cnnModel')
])

app.config['suppress_callback_exceptions'] = True

@app.callback(
    [
        Output('deepfaceModel', 'figure'),
        Output('ferModel', 'figure'),
        Output('cnnModel', 'figure'),
        Output('show-all-btn', 'n_clicks'),
    ],
    [
        Input('show-all-btn', 'n_clicks')
    ]
)
def update_graph(show_all_clicks):
    emotion_chosen = ''

    if show_all_clicks is None:
        show_all_clicks = 0

    if show_all_clicks > 0:
        emotion_chosen = ''
        show_all_clicks -= 1

    fig = go.Figure()
    fig2 = go.Figure()
    fig3 = go.Figure()

    if emotion_chosen == '':
        for col in column_names:
            fig.add_trace(go.Bar(
                x=[col],
                y=[deepfaceCSV.loc[0][col]],
                marker=dict(color="#00cc00"),
            ))
            fig2.add_trace(go.Bar(
                x=[col],
                y=[ferCSV.loc[0][col]],
                marker=dict(color="royalblue"),
            ))
            fig3.add_trace(go.Bar(
                x=[col],
                y=[cnnCSV.loc[0][col]],
                marker=dict(color="gold"),
            ))

    for i, emotion in enumerate(column_names):
        fig.data[i].name = emotion
        fig2.data[i].name = emotion
        fig3.data[i].name = emotion

    fig.update_layout(
        title='Deepface Model',
        xaxis=dict(gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray'),
        plot_bgcolor='#333',
        paper_bgcolor='#333',
        title_font=dict(color='white', size=20),
        title_x=0.5,
        legend=dict(
            orientation='h',
            x=0.5,
            y=-0.3,
            xanchor='center',
            yanchor='top',
            tracegroupgap=100,
            itemsizing='constant',
            itemwidth=80,
            font=dict(color="white")
        )
    )
    fig2.update_layout(
        title='Fer Model',
        xaxis=dict(gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray'),
        plot_bgcolor='#333',
        paper_bgcolor='#333',
        title_font=dict(color='white', size=20),
        title_x=0.5,
        legend=dict(
            orientation='h',
            x=0.5,
            y=-0.3,
            xanchor='center',
            yanchor='top',
            tracegroupgap=100,
            itemsizing='constant',
            itemwidth=80,
            font=dict(color="white")
        )
    )
    fig3.update_layout(
        title='CNN Model',
        xaxis=dict(gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray'),
        plot_bgcolor='#333',
        paper_bgcolor='#333',
        title_font=dict(color='white', size=20),
        title_x=0.5,
        legend=dict(
            orientation='h',
            x=0.5,
            y=-0.3,
            xanchor='center',
            yanchor='top',
            tracegroupgap=100,
            itemsizing='constant',
            itemwidth=80,
            font=dict(color="white")
        )
    )

    fig2.update_traces(marker=dict(color="royalblue"))
    fig3.update_traces(marker=dict(color="gold"))

    fig.update_xaxes(title='Emotion', color='white')
    fig.update_yaxes(title='Percentage', color='white')
    fig2.update_xaxes(title='Emotion', color='white')
    fig2.update_yaxes(title='Percentage', color='white')
    fig3.update_xaxes(title='Emotion', color='white')
    fig3.update_yaxes(title='Percentage', color='white')
    return fig, fig2, fig3, show_all_clicks

if __name__ == '__main__':
    app.run_server(debug=True)
