#%%
from datetime import datetime
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

def GetPriceData_Yfinance(
        symbol, 
        start_time, 
        end_time, 
        timeframe,
        ):
    OHLC = yf.Ticker(symbol).history(
        # [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
        interval=timeframe,
        # period=Duration,
        start = start_time,
        end = end_time,
        )
    return OHLC

with st.sidebar:
    st.title('Inputs')
    symbol = st.text_input('Symbol', 'MSFT')
    start_time = st.date_input('Start time', datetime(2000, 1, 1))
    end_time = st.date_input('End time', datetime.now())
    timeframe = st.select_slider('Timeframe', options=['1m', '5m', '15m', '30m', '1h', '1d'], value='1d')
    Nstates = st.slider('Number of states', 5, 100, 50)
    feature = st.selectbox('Feature', ['log_return', 'EMA_log_return', 'volatility', 'EMA_volatility', 'log_volatility', 'EMA_log_volatility'])
endTime = datetime.now()

#%    GET PRICE DATA
data = GetPriceData_Yfinance(
    symbol = symbol,
    start_time = start_time,
    end_time = end_time,
    timeframe = timeframe,
    )

if data.empty:
    st.write('No data available for the selected symbol and timeframe')
    st.stop()

#%    MARKOV CHAIN
data['return'] = data['Close'] / data['Close'].shift(1) - 1
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
data['EMA_log_return'] = data['log_return'].ewm(span=20, adjust=False).mean()
data['volatility'] = data['log_return'].rolling(20).std()
data['EMA_volatility'] = data['volatility'].ewm(span=20, adjust=False).mean()
data['log_volatility'] = np.log(data['volatility'])
data['EMA_log_volatility'] = data['log_volatility'].ewm(span=20, adjust=False).mean()

feature_array = data[feature].to_numpy()   # convert to numpy array

p0 = np.nanmin(feature_array)
p1 = np.nanmax(feature_array)

# calculate the thresholds for each state based on Nstates and pctChangeState
thds = np.zeros(Nstates)
thds = np.linspace(p0, p1, Nstates)

groups = np.zeros(len(data['Close']))

for i in range(1, len(data)):
    for j in range(1, len(thds)):
        if feature_array[i] > thds[j-1] and feature_array[i] <= thds[j]:
            groups[i] = j  # state 0 is the first state
            break

# remove the first data from the groups
groups = groups[1:]
groups = groups.astype(int)

# calculate the probabilities of transition from one state to another
transitions = np.zeros((Nstates-1, Nstates-1))
for i in range(1, len(groups)):
    transitions[groups[i-1]-1, groups[i]-1] += 1

# replace 0 with np.nan
transitions[transitions == 0] = np.nan

transitions = np.log(transitions)

x = thds
y = thds


fig = go.Figure()
fig.add_trace(go.Heatmap(   
    x=x,
    y=y,    
    z=transitions,
    colorscale='Viridis',
    colorbar=dict(title='log(Count)'),
))
fig.update_layout(
    title = f'Transition matrix for {symbol} based on {timeframe} data and {feature}',
    xaxis_title = 'From State',
    yaxis_title = 'To State',
    template = 'seaborn',
    xaxis=dict(scaleanchor="y", scaleratio=1),
)

# add line y=x
fig.add_trace(go.Scatter(
    x=x,
    y=x,
    mode='lines',
    line=dict(color='black', width=1, dash='dash'),
    name='y=x',
))

fig.update_xaxes(
    zeroline = True,
    zerolinewidth = 1,
    zerolinecolor = 'Black',
)
fig.update_yaxes(
    zeroline = True,
    zerolinewidth = 1,
    zerolinecolor = 'Black',
)

# fig.show()
st.title('Markov Chain')
st.write("""
    This simple tool calculates the transition matrix for a given feature of a symbol.
    The transition matrix is a matrix that shows the probability of transitioning from one state to another.
    It can be formulated as:
         
    $$ 
    T = \\begin{bmatrix} 
    P_{11} & P_{12} & \\cdots & P_{1n} \\\\
    P_{21} & P_{22} & \\cdots & P_{2n} \\\\
    \\vdots & \\vdots & \\ddots & \\vdots \\\\ 
    P_{n1} & P_{n2} & \\cdots & P_{nn} 
    \\end{bmatrix} 
    $$
         
    where $P_{ij}$ is the probability of transitioning from state $i$ to state $j$.
    The transition matrix is based on the assumption that the future state depends only on the current state.
    This is typically called a Markov Chain.
    
    The transition matrix can be visualized as a heatmap where the x-axis represents the current state and the y-axis represents the future state.
    """)
st.plotly_chart(fig)
