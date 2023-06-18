import pandas as pd
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import plotly.graph_objects as go

df = pd.read_csv("models/deepface/csv_files/emotions_deepface.csv")

app = Dash(__name__, assets_folder="assets")
app.layout = html.Div(
    [
        html.Div(children="Facial Expressions Results", className="title"),
        html.Hr(),
        dcc.RadioItems(
            options=[{"label": key, "value": key} for key in df.columns],
            value="",
            id="controls-and-radio-item",
            className="radio-buttons",
        ),
        dcc.Graph(figure={}, id="deepfaceModel", className="deepfaceModel"),
        dcc.Graph(figure={}, id="ferModel", className="ferModel"),
    ]
)


@callback(
    [
        Output(component_id="deepfaceModel", component_property="figure"),
        Output(component_id="ferModel", component_property="figure"),
    ],
    [Input(component_id="controls-and-radio-item", component_property="value")],
)
def update_graph(emotion_chosen):
    colors = ["gold", "blue", "#28fc03", "red", "#000", "purple", "pink"]
    fig = go.Figure()
    fig2 = go.Figure()
    if emotion_chosen == "":
        for col in df.columns:
            fig.add_trace(
                go.Bar(
                    x=[col],
                    y=[df.loc[0][col]],
                    marker=dict(color=colors[2]),
                )
            )
            fig2.add_trace(
                go.Bar(
                    x=[col],
                    y=[df.loc[0][col]],
                    marker=dict(color=colors[2]),
                )
            )
    else:
        for col in df.columns:
            if col == emotion_chosen:
                fig.add_trace(
                    go.Bar(
                        x=[col],
                        y=[df.loc[0][col]],
                        marker=dict(color=colors[2]),
                    )
                )
                fig2.add_trace(
                    go.Bar(
                        x=[col],
                        y=[df.loc[0][col]],
                        marker=dict(color=colors[2]),
                    )
                )
            else:
                fig.add_trace(
                    go.Bar(
                        x=[col],
                        y=[0],
                        marker=dict(color="rgba(0,0,0,0)"),
                    )
                )
                fig2.add_trace(
                    go.Bar(
                        x=[col],
                        y=[0],
                        marker=dict(color="rgba(0,0,0,0)"),
                    )
                )

    fig.update_layout(
        title="Deepface Model",
        xaxis=dict(gridcolor="lightgray"),
        yaxis=dict(gridcolor="lightgray"),
        plot_bgcolor="#333",
        paper_bgcolor="#333",
        title_font=dict(color="white", size=20),
        title_x=0.5,
    )
    fig2.update_layout(
        title="Deepface Model",
        xaxis=dict(gridcolor="lightgray"),
        yaxis=dict(gridcolor="lightgray"),
        plot_bgcolor="#333",
        paper_bgcolor="#333",
        title_font=dict(color="white", size=20),
        title_x=0.5,
    )
    fig.update_xaxes(title="Emotion", color="white")
    fig.update_yaxes(title="Percentage", color="white")
    fig2.update_xaxes(title="Emotion", color="white")
    fig2.update_yaxes(title="Percentage", color="white")
    return fig, fig2


if __name__ == "__main__":
    app.run_server(debug=True)
