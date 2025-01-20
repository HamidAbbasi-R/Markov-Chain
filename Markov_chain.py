#%%
from datetime import datetime
import pandas as pd
import pandas_ta as ta
import MetaTrader5 as mt5
import numpy as np
from datetime import datetime
import streamlit as st
# import pandas_ta as ta
import plotly.graph_objects as go
import plotly.express as px
# render on vscode
# import plotly.io as pio
# pio.renderers.default = 'vscode'

# mt5.initialize()

def GetPriceData(
        symbol, 
        endTime = datetime.now(),
        timeframe = 'M5',
        Nbars = 1000,
        source = 'MT5',
        indicators_dict = {
            'ATR':      False,
            'ADX':      False,
            'RSI':      False,
        },
        MA_period = 20,
        ):
    
    if source=='MT5':
        # move the hour forward by 2 hours 
        endTime = endTime + pd.DateOffset(hours=2)

        # if Nbars is larger than 99999, get the data in chunks
        rates = pd.DataFrame()  # Initialize an empty DataFrame
        while Nbars > 0:
            Nbars_chunk = min(Nbars, 200000)
            Nbars -= Nbars_chunk

            rates_chunk = mt5.copy_rates_from(
                symbol, 
                ConvertTimeFrametoMT5(timeframe), 
                endTime,
                Nbars_chunk,
            )

            # convert to pandas DataFrame
            rates_chunk = pd.DataFrame(rates_chunk)

            # Add the retrieved chunk to the overall list
            rates = pd.concat([rates, rates_chunk], ignore_index=True)

            # Update endTime to the last time of the retrieved data
            endTime = rates_chunk['time'][0]  # Assuming the data is sorted in reverse chronological order
            
            # convert the endTime from int64 to datetime
            endTime = pd.to_datetime(endTime, unit='s')
            
        # convert times to UTC+1
        rates['time']=pd.to_datetime(rates['time'], unit='s')
        rates['time'] = rates['time'] + pd.DateOffset(hours=-2)

        rates['hour'] = rates['time'].dt.hour

        rates['MA_close'] = rates['close'].rolling(MA_period).mean()
        rates['EMA_close'] = rates['close'].ewm(span=MA_period, adjust=False).mean()

        # remove nans
        rates = rates.dropna()
        rates.rename(columns={'tick_volume': 'volume'}, inplace=True)
        rates['MA_volume'] = rates['volume'].rolling(MA_period).mean()
        rates['EMA_volume'] = rates['volume'].ewm(span=MA_period, adjust=False).mean()
        
        rates['log_volume'] = np.log(rates['volume'])
        rates['MA_log_volume'] = rates['log_volume'].rolling(MA_period).mean()
        rates['EMA_log_volume'] = rates['log_volume'].ewm(span=MA_period, adjust=False).mean()
        
        rates['log_return'] = np.log(rates['close'] / rates['close'].shift(1))
        rates['MA_log_return'] = rates['log_return'].rolling(MA_period).mean()       
        rates['EMA_log_return'] = rates['log_return'].ewm(span=MA_period, adjust=False).mean()
        
        rates['volatility'] = rates['log_return'].rolling(MA_period).std()
        rates['MA_volatility'] = rates['volatility'].rolling(MA_period).std()   
        rates['EMA_volatility'] = rates['volatility'].ewm(span=MA_period, adjust=False).std()
        
        rates['log_volatility'] = np.log(rates['volatility'])
        rates['MA_log_volatility'] = rates['log_volatility'].rolling(MA_period).mean()
        rates['EMA_log_volatility'] = rates['log_volatility'].ewm(span=MA_period, adjust=False).mean()
        
        rates['MA_volume'] = rates['volume'].rolling(MA_period).mean()
        rates['EMA_volume'] = rates['volume'].ewm(span=MA_period, adjust=False).mean()
        
        rates['upward'] = (rates['log_return'] > 0).astype(int)
            

        if indicators_dict['ATR']:
            rates['ATR'] = ta.atr(rates['high'], rates['low'], rates['close'], length=MA_period)
            
        if indicators_dict['ADX']:
            ADX = ta.adx(rates['high'], rates['low'], rates['close'], length=MA_period)
            rates['ADX'] = ADX[f'ADX_{MA_period}']

        if indicators_dict['RSI']:
            rates['RSI'] = ta.rsi(rates['close'], length=MA_period)
      
        return rates
    
    elif source=='yfinance':
        startTime = get_start_time(endTime, timeframe, Nbars)
        # convert the symbol to the format required by yfinance
        # AVAILABLE ASSETS
        # 'USDJPY=X' , 'USDCHF=X' , 'USDCAD=X', 
        # 'EURUSD=X' , 'GBPUSD=X' , 'AUDUSD=X' , 'NZDUSD=X', 
        # 'BTC-USD', 'ETH-USD', 'BNB-USD', 
        # 'XRP-USD', 'ADA-USD', 'SOL-USD', 'DOT-USD'
        if symbol[:3] in ['BTC', 'ETH', 'XRP', 'BNB', 'ADA', 'DOGE', 'DOT', 'SOL']:
            symbol = symbol[:3] + '-' + symbol[3:]
        else:
            symbol = symbol + '=X'
            # pass
        # convert timeframe to yfinance format
        timeframe = ConvertTimeFrametoYfinance(timeframe)
        rates = GetPriceData_Yfinance(symbol, startTime, endTime, timeframe)
        # change keys name from Close, Open, High, Low to close, open, high, low
        rates = rates.rename(columns={'Close':'close', 'Open':'open', 'High':'high', 'Low':'low'})
        # change keys name from Date to time
        rates['time'] = rates.index
        return rates

def ConvertTimeFrametoYfinance(timeframe):
    timeframes = {
        'M1': '1m',
        'M5': '5m',
        'M15': '15m',
        'M30': '30m',
        'H1': '1h',
        'H4': '4h',
        'D1': '1d',
        'W1': '1wk',
        'MN1': '1mo'
    }
    return timeframes.get(timeframe, 'Invalid timeframe')

def ConvertTimeFrametoMT5(timeframe):
    timeframes = {
        'M1': mt5.TIMEFRAME_M1,
        'M2': mt5.TIMEFRAME_M2,
        'M3': mt5.TIMEFRAME_M3,
        'M4': mt5.TIMEFRAME_M4,
        'M5': mt5.TIMEFRAME_M5,
        'M6': mt5.TIMEFRAME_M6,
        'M10': mt5.TIMEFRAME_M10,
        'M12': mt5.TIMEFRAME_M12,
        'M15': mt5.TIMEFRAME_M15,
        'M20': mt5.TIMEFRAME_M20,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H2': mt5.TIMEFRAME_H2,
        'H3': mt5.TIMEFRAME_H3,
        'H4': mt5.TIMEFRAME_H4,
        'H6': mt5.TIMEFRAME_H6,
        'H8': mt5.TIMEFRAME_H8,
        'H12': mt5.TIMEFRAME_H12,
        'D1': mt5.TIMEFRAME_D1,
        'W1': mt5.TIMEFRAME_W1,
        'MN1': mt5.TIMEFRAME_MN1
    }
    return timeframes.get(timeframe, 'Invalid timeframe')

def GetPriceData_Yfinance(
        symbol, 
        start_time, 
        end_time, 
        timeframe,
        ):
    import yfinance as yf
    OHLC = yf.Ticker(symbol).history(
                # [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
                interval=timeframe,
                # period=Duration,
                start = start_time,
                end = end_time,
            )
    return OHLC

def get_start_time(
        endTime, 
        timeframe, 
        Nbars,
        ):
    import re
    from datetime import timedelta
    def get_time_per_bar(timeframe):
    # Use regex to capture the numeric part and the unit
        match = re.match(r'([A-Za-z]+)(\d+)', timeframe)
        if not match:
            raise ValueError(f"Invalid timeframe format: {timeframe}")
    
        unit = match.group(1).upper()  # Get the letter part (M, H, D)
        value = int(match.group(2))    # Get the numeric part

        # Convert unit to appropriate timedelta
        if unit == 'M':  # Minutes
            return timedelta(minutes=value)
        elif unit == 'H':  # Hours
            return timedelta(hours=value)
        elif unit == 'D':  # Days
            return timedelta(days=value)
        else:
            raise ValueError(f"Unsupported timeframe unit: {unit}")

    # Get time per bar based on the timeframe
    time_per_bar = get_time_per_bar(timeframe)

    # Calculate total time to subtract
    total_time = time_per_bar * Nbars

    # Calculate the startTime
    startTime = endTime - total_time

    return startTime

with st.sidebar:
    st.title('Inputs')
    symbol = st.text_input('Symbol (for stocks add .US to the end eg. TSLA.US)', 'EURUSD')
    timeframe = st.select_slider('Timeframe', options=['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'], value='D1')
    fromNow = st.checkbox('From now', value=True)
    Nbars = st.slider('Number of bars', 1000, 100000, 10000)
    Nstates = st.slider('Number of states', 5, 100, 50)
    feature = st.selectbox('Feature', ['log_return', 'EMA_log_return', 'volatility', 'EMA_volatility', 'log_volatility', 'EMA_log_volatility'])
endTime = datetime.now()


if fromNow:
    endTime = datetime.now()

#%    GET PRICE DATA
data = GetPriceData(
    symbol = symbol, 
    endTime = endTime, 
    timeframe = timeframe, 
    Nbars = Nbars, 
    # source='MT5',
    source='yfinance',
    )

#%    MARKOV CHAIN
data['return'] = data['close'] / data['close'].shift(1) - 1
data['log_return'] = np.log(data['close'] / data['close'].shift(1))
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

groups = np.zeros(len(data['close']))

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
st.write(f'Transition matrix for {symbol} based on {timeframe} data and {feature}')
st.plotly_chart(fig)
