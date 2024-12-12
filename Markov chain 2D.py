#%%     LIBRARIES
import functions as fns
import numpy as np
from datetime import datetime
import pandas_ta as ta
import plotly.graph_objects as go
# render on vscode
import plotly.io as pio
pio.renderers.default = 'vscode'

#%%    CONSTANTS
symbol = 'EURUSD'
timeframe = "M5"
endTime = datetime(        # in LONDON time
    year = 2024, 
    month = 10, 
    day = 16, 
    hour = 10,
    minute = 0,
    second = 0,
)
Nbars = 2000

Nstates1 = 10
Nstates2 = 10

binSize1 = 0.01       # bin size for each state 1
binSize2 = 0.00001       # bin size for each state 2
#%%    GET PRICE DATA
data = fns.GetPriceData(symbol, endTime, timeframe, Nbars)

#%%    MARKOV CHAIN
# first data
data['Pct_Change'] = data['close'].pct_change() * 100
pctChange = data['Pct_Change'].to_numpy()   # convert to numpy array
Data1 = pctChange


# second data
data['sma'] = ta.sma(data['close'], 14)
diffSMAClose = data['close'] - data['sma']
diffSMAClose = diffSMAClose.to_numpy()  # convert to numpy array
Data2 = diffSMAClose


# calculate the thresholds for each state based on Nstates and pctChangeState
thds1 = np.zeros(Nstates1*2+3)
thds1[0] = -np.inf
thds1[-1] = np.inf
thds1[1:-1] = np.linspace(-binSize1*Nstates1, binSize1*Nstates1, Nstates1*2+1)

thds2 = np.zeros(Nstates2*2+3)
thds2[0] = -np.inf
thds2[-1] = np.inf
thds2[1:-1] = np.linspace(-binSize2*Nstates2, binSize2*Nstates2, Nstates2*2+1)

groups = [None] * len(data) 

for i in range(1, len(data)):
    for j in range(1, len(thds1)):
        if Data1[i] > thds1[j-1] and Data1[i] <= thds1[j]:
            for k in range(1, len(thds2)):
                if Data2[i] > thds2[k-1] and Data2[i] <= thds2[k]:
                    groups[i] = [int(j), int(k)]  # state 0 is the first state
                    break  # exit the innermost loop once the condition is met
            break  # exit the second loop once the condition is met

# remove the first data from the groups
groups = groups[14:]


# calculate the probabilities of transition from one state to another
lenMatrix = (Nstates1*2+2) * (Nstates2*2+2)
transitions = np.zeros((lenMatrix, lenMatrix))
for i in range(1, len(groups)):
    transitions[(groups[i-1][0]-1) * (Nstates2*2+2) + groups[i-1][1]-1, (groups[i][0]-1) * (Nstates2*2+2) + groups[i][1]-1] += 1

# replace 0 with np.nan
transitions[transitions == 0] = np.nan
# for i in range(transitions.shape[0]):
#     transitions[i] /= np.sum(transitions[i])

# log scale
transitions = np.log(transitions)

# x = thds
# y = thds

fig = go.Figure(data=go.Heatmap(
    z = transitions,
    # x = x,
    # y = y,
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
fig.show()