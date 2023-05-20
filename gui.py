from dash import Dash, html, dash_table, dcc, callback, Output, Input
import plotly.graph_objects as go

data = [{'emotion': {'angry': 46.04460000991821, 'disgust': 9.36625212943909e-05, 'fear': 8.291973918676376, 'happy': 0.0001523979790363228, 'sad': 38.99666368961334, 'surprise': 4.050834547797422e-05, 'neutral': 6.666478514671326}, 'dominant_emotion': 'angry'}]

app = Dash(__name__, assets_folder='assets')
app.layout = html.Div([
    html.Div(children='Facial Expressions Results', className='title'),
    html.Hr(),
    dcc.RadioItems(
        options=[{'label': key, 'value': key} for key in data[0]['emotion'].keys()],
        value='angry',
        id='controls-and-radio-item',
        className='radio-buttons'
    ),
    dcc.Graph(figure={}, id='controls-and-graph', className='controls-and-graph')
])

@callback(
    Output(component_id='controls-and-graph', component_property='figure'),
    Input(component_id='controls-and-radio-item', component_property='value')
)
def update_graph(emotion_chosen):
    colors = ['gold', 'blue', '#28fc03', 'red', '#000', 'purple', 'pink']
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(data[0]['emotion'].keys()),
        y=list(data[0]['emotion'].values()),
        marker=dict(
            color=colors[2],
        ),
    ))
    fig.update_layout(
        title=f'Emotion Distribution for {emotion_chosen}',
        xaxis=dict(gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray'),
        plot_bgcolor='#333',
        paper_bgcolor='#333',
        title_font=dict(color='white', size=20),
        title_x=0.5

    )
    fig.update_xaxes(title='Emotion', color='white')
    fig.update_yaxes(title='Percentage', color='white')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
