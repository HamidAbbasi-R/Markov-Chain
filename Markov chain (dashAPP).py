#%%
from dash import Dash, dcc, html, Input, Output
from datetime import datetime
#%     LIBRARIES
import functions as fns
import numpy as np
from datetime import datetime
# import pandas_ta as ta
import plotly.graph_objects as go
# render on vscode
# import plotly.io as pio
# pio.renderers.default = 'vscode'


# Initialize the Dash app
app = Dash(__name__)

# Define the app layout
app.layout = html.Div([
    html.Div([
        dcc.Input(id='symbol', type='text', placeholder='Enter Symbol'),
        
        dcc.Dropdown(id='timeframe',
            options=[
                {'label': '1 Minute (M1)', 'value': 'M1'},
                {'label': '5 Minutes (M5)', 'value': 'M5'},
                {'label': '15 Minutes (M15)', 'value': 'M15'},
                {'label': '30 Minutes (M30)', 'value': 'M30'},
                {'label': '1 Hour (H1)', 'value': 'H1'},
                {'label': '4 Hours (H4)', 'value': 'H4'},
                {'label': '1 Day (D1)', 'value': 'D1'},
            ],
            placeholder='Select Timeframe'
        ),
        
        dcc.DatePickerSingle(id='endTime',
            date=datetime.now()
        ),
        
        dcc.Checklist(id='fromNow',
            options=[{'label': 'From Now', 'value': 'Yes'}],
            value=[]
        ),
        
        dcc.Input(id='Nbars', type='number', placeholder='Enter Number of Bars'),
        
        dcc.Input(id='Nstates', type='number', placeholder='Enter Number of States'),
        
        dcc.Input(id='pctChangeState', type='number', placeholder='Enter Percentage Change State'),
        
        html.Button('Submit', id='submit-btn', n_clicks=0),
    ], style={'width': '15%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    
    html.Div([
        dcc.Graph(id='heatmap', style={'width': '100%','height': '100vh', 'display': 'inline-block'}),
        ], 
        style={
            'display': 'inline-block',
            'verticalAlign': 'top'}),
])

# Callback to update the heatmap figure
@app.callback(
    Output('heatmap', 'figure'),
    [Input('submit-btn', 'n_clicks')],
    [
        Input('symbol', 'value'),
        Input('timeframe', 'value'),
        Input('endTime', 'value'),
        Input('fromNow', 'value'),
        Input('Nbars', 'value'),
        Input('Nstates', 'value'),
        Input('pctChangeState', 'value'),
    ]
)
def update_heatmap(n_clicks, symbol, timeframe, endTime, fromNow, Nbars, Nstates, pctChangeState):
    # Call the test function to generate the figure
    if n_clicks > 0:
        return get_markov_chain_plot(symbol, timeframe, endTime, fromNow, Nbars, Nstates, pctChangeState)
    return {}

def get_markov_chain_plot(
        symbol = 'EURUSD', 
        timeframe = 'H1', 
        endTime = datetime.now(), 
        fromNow = True, 
        Nbars = 1000, 
        Nstates = 100, 
        pctChangeState = 0.1,
        ):

    if fromNow:
        endTime = datetime.now()

    #%    GET PRICE DATA
    data = fns.GetPriceData(symbol, endTime, timeframe, Nbars, source='MT5')

    #%    MARKOV CHAIN
    data['Pct_Change'] = np.log(data['close']).diff() * 100
    pctChange = data['Pct_Change'].to_numpy()   # convert to numpy array

    # calculate the thresholds for each state based on Nstates and pctChangeState
    thds = np.zeros(Nstates*2+3)
    thds[0] = -np.inf
    thds[-1] = np.inf
    thds[1:-1] = np.linspace(-pctChangeState*Nstates, pctChangeState*Nstates, Nstates*2+1)

    groups = np.zeros(len(data['close']))

    for i in range(1, len(data)):
        for j in range(1, len(thds)):
            if pctChange[i] > thds[j-1] and pctChange[i] <= thds[j]:
                groups[i] = j  # state 0 is the first state
                break

    # remove the first data from the groups
    groups = groups[1:]
    groups = groups.astype(int)

    # calculate the probabilities of transition from one state to another
    transitions = np.zeros((Nstates*2+2, Nstates*2+2))
    for i in range(1, len(groups)):
        transitions[groups[i-1]-1, groups[i]-1] += 1

    # replace 0 with np.nan
    transitions[transitions == 0] = np.nan
    # for i in range(transitions.shape[0]):
    #     transitions[i] /= np.sum(transitions[i])

    # log scale
    transitions = np.log(transitions)

    x = thds
    y = thds

    fig = go.Figure(data=go.Heatmap(
        z = transitions,
        x = x,
        y = y,
        colorscale = 'Viridis'
    ))
    fig.update_layout(
        title = 'Transition matrix',
        xaxis_title = 'From State',
        yaxis_title = 'To State',
        # xaxis = dict(tickvals = np.arange(Nstates*2+3)),
        # yaxis = dict(tickvals = np.arange(Nstates*2+3)),
        template = 'plotly_white'
    )
    # same scale for x and y axis
    fig.update_layout(xaxis=dict(scaleanchor="y", scaleratio=1))
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)
