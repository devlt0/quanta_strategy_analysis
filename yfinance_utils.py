from datetime      import datetime, date, timedelta

import yfinance  as yf
import pandas    as pd
import pandas_ta as ta
import numpy as np
import traceback

def add_supertrend_indicator(data: pd.DataFrame, ticker:str, super_time_period=10, multiplier=3) -> pd.DataFrame:
    """
    SuperTrend: Combines trend and volatility to identify potential breakouts.
    - Uses ATR to adapt to volatility
    - When price crosses above SuperTrend line = bullish signal
    """

    supertrend = ta.supertrend(
        high   = data[('High', ticker)],
        low    = data[('Low', ticker)],
        close  = data[('Close', ticker)],
        length = super_time_period,
        multiplier=multiplier
    )

    #data.ta.supertrend(length=10, multiplier=4.0, append=True)  # doesn't work with multiindexing

    data['SuperTrend']           = supertrend[f'SUPERT_{super_time_period}_{multiplier}']
    data['SuperTrend_Direction'] = supertrend[f'SUPERTd_{super_time_period}_{multiplier}']
    data['SuperTrend_says_buy']  = data['SuperTrend_Direction'] == 1
    return data

def add_keltner_channels(data: pd.DataFrame, ticker:str, keltner_time_period=20, multiplier=2.0) -> pd.DataFrame:
    """
    Keltner Channels: Similar to Bollinger Bands but uses ATR for volatility
    - When price breaks above upper Keltner Channel = strong momentum
    - More stable than Bollinger Bands in trending markets
    - Good for confirming strength of breakouts
    """
    keltner = ta.kc(
        high   = data[('High', ticker)],
        low    = data[('Low', ticker)],
        close  = data[('Close', ticker)],
        length = keltner_time_period,
        scalar = multiplier
    )

    data['KC_Upper']  = keltner[f'KCUe_{keltner_time_period}_{float(multiplier)}']
    data['KC_Lower']  = keltner[f'KCLe_{keltner_time_period}_{float(multiplier)}']
    data['KC_Middle'] = keltner[f'KCBe_{keltner_time_period}_{float(multiplier)}']
    data['KC_says_buy'] = data[('Close', ticker)] > data['KC_Upper']
    return data

def add_vwap_indicator(data: pd.DataFrame, ticker:str, vwap_time_period=14) -> pd.DataFrame:
    """
    VWAP (Volume Weighted Average Price): Combines price and volume
    - Price above VWAP = bullish
    - High volume above VWAP = strong breakout confirmation
    - Institutional traders often use VWAP for entries
    """
    data['VWAP'] = ta.vwap(
        high   = data[('High', ticker)],
        low    = data[('Low', ticker)],
        close  = data[('Close', ticker)],
        volume = data[('Volume', ticker)],
        anchor='D'
    )

    data['VWAP_says_buy'] = (data[('Close', ticker)] > data['VWAP']) & (data[('Volume', ticker)] > data[('Volume', ticker)].rolling(window=vwap_time_period, min_periods=vwap_time_period, center=False).mean())
    return data

def add_momentum_indicators(data: pd.DataFrame, ticker:str) -> pd.DataFrame:
    """
    Multiple momentum indicators combined:
    - ROC (Rate of Change): Measures momentum
    - PPO (Percentage Price Oscillator): Similar to MACD but in percentage terms
    - CMF (Chaikin Money Flow): Volume-weighted momentum
    """
    # Rate of Change
    data['ROC'] = ta.roc(data[('Close', ticker)], length=9)

    # Percentage Price Oscillator
    ppo = ta.ppo(data[('Close', ticker)])
    data['PPO'] = ppo[f'PPO_{12}_{26}_{9}']
    data['PPO_Signal'] = ppo[f'PPOh_{12}_{26}_{9}']

    # Chaikin Money Flow
    data['CMF'] = ta.cmf(
        high   = data[('High', ticker)],
        low    = data[('Low', ticker)],
        close  = data[('Close', ticker)],
        volume = data[('Volume', ticker)],
        length=20
    )

    # Combined momentum buy signal
    data['Momentum_says_buy'] = (
        (data['ROC'] > 0) &  # Positive momentum
        (data['PPO'] > data['PPO_Signal']) &  # Bullish PPO crossover
        (data['CMF'] > 0)  # Positive money flow
    )
    return data

def add_volume_profile(data: pd.DataFrame, ticker:str, window_time_period=20) -> pd.DataFrame:
    """
    Volume Profile Analysis:
    - Identifies high volume nodes and potential breakout levels
    - Looks for price moving above high volume areas with strong volume
    - Uses relative volume analysis
    """
    # Calculate volume profile metrics
    data['Volume_SMA'] = data[('Volume', ticker)].rolling(window=window_time_period, min_periods=window_time_period, center=False).mean()
    data['Volume_StdDev'] = data[('Volume', ticker)].rolling(window=window_time_period, min_periods=window_time_period, center=False).std()
    data['Relative_Volume'] = data[('Volume', ticker)] / data['Volume_SMA']

    # Calculate price levels with historically high volume
    data['High_Volume_Level'] = data[('Close', ticker)].rolling(window=window_time_period, min_periods=window_time_period, center=False).mean()

    # Volume breakout conditions
    data['Volume_Profile_says_buy'] = (
        (data[('Close', ticker)] > data['High_Volume_Level']) &  # Price above high volume level
        (data['Relative_Volume'] > 1.5) &  # Higher than normal volume
        (data[('Close', ticker)] > data[('Close', ticker)].shift(1))  # Price increasing
    )
    return data

def combine_breakout_signals(data: pd.DataFrame, ticker:str) -> pd.DataFrame:
    """
    Creates various combinations of indicators for stronger breakout signals
    """
     #st.session_state.data['IsBase+MACD+BB+OBVIndicatorsBreakoutDay'] = st.session_state.data.apply(lambda zzz: zzz["IsVolThresholdMet"] & zzz["IsDailyPChangeMet"]  & zzz["MACD_says_buy"]  & zzz["BB_says_buy"]  & zzz["OBV_says_buy"] , axis=1 )
    data['Momentum_Volume_Breakout'] = data.apply(lambda mvb:
        mvb['Momentum_says_buy'] &
        mvb['Volume_Profile_says_buy'] &
        mvb['VolumeChange'] >= 1.0,
        axis=1)

    #st.write("mvb")


    data['Trend_Confirmation_Breakout'] = data.apply(lambda tcb:
        #tcb['SuperTrend_says_buy'] &
        tcb['MACD_says_buy'] &
        tcb['RSI_says_buy'],
        axis=1)

    #st.write("tcb")

    data['Volatility_Breakout'] = data.apply(lambda vb:
        vb['KC_says_buy'] &
        vb['BB_says_buy'] &
        vb['ATR_says_buy'],
        axis=1)

    #st.write("vb")

    data['Volume_Price_Breakout'] = data.apply(lambda vpb:
        vpb['VWAP_says_buy'] &
        vpb['Volume_Profile_says_buy'] &
        vpb['OBV_says_buy'],
        axis=1)

    #st.write("vpb")

    data['Ultimate_Breakout'] = data.apply(lambda ub:
        #(ub[('Close', ticker)] > ub['BB_upper']) &    # Price above upper Bollinger Band
        ub['IsBreakoutDay'] &       # Confirmed uptrend
        ub['Momentum_says_buy'] &         # Strong momentum
        ub['Volume_Profile_says_buy'] &   # Volume confirmation
        ub['KC_says_buy'],                # Volatility breakout
        axis=1)

    #st.write("ub")
    return data


# --- Data Fetching and Processing ---
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def check_if_weekend(given_date:str="")->bool:
    # good idea, not the real limitation
    # real limitation is whether or not index exists for given date
    # presumption is market closed on specific holidays or for other similar reasons on weekdays
    last_weekday_index = 4
    date_is_weekend = True if given_date.weekday() > last_weekday_index else False
    return date_is_weekend


def get_datetimestamp_str(str_format:str='%Y-%m-%d__%H-%M-%S')->str:
    """
    '%Y-%m-%d__%H-%M-%S-%f'
    Year-month-day__Hour-Minute-Seconds-Microseconds
    """
    #'%H-%M-%S_%d-%m-%Y'
    current_time = datetime.now()
    formatted_time = current_time.strftime(str_format)
    return formatted_time


def gen_ticker_timestamp_suffix_str(ticker, datetime_format:str='%Y-%m-%d__%H-%M-%S')->str:
    suffix = f"_{ticker}_{get_datetimestamp_str(datetime_format)}"
    return suffix

def calculate_returns(data, ticker, buy_dates, holding_period):
    results = []
    all_dates = data["Date"].to_list()
    running_total_return = 0
    for buy_date_str in buy_dates:
        try:
            buy_date = pd.to_datetime(buy_date_str) #Convert to datetime object here
            skip_iter = 0
            while buy_date not in all_dates:
                skip_iter += 1
                buy_date = buy_date + pd.Timedelta(days= skip_iter)

            #print(f"\n\n{buy_date}\n{type(buy_date)}")
            buy_price = data.loc[data["Date"] == buy_date, 'Close'].squeeze()
            #print(buy_price)
            # age_value = df.loc[df['Name'] == name_to_lookup, 'Age'].iloc[0]
            sell_date = buy_date + pd.Timedelta(days=holding_period)
            skip_iter = 0
            # ensure we have a valid end date- this could muck the hold date bit and be confusingly mixed in
            while sell_date not in all_dates:
                skip_iter += 1
                sell_date = buy_date + pd.Timedelta(days=holding_period + skip_iter)

            sell_price =  data.loc[data["Date"] == sell_date, 'Close'].squeeze()   #.iloc[0] #data['Close'][sell_date]
            return_percent = ((sell_price - buy_price) / buy_price) * 100
            results.append({'Buy Date': buy_date, 'Sell Date': sell_date, 'Return (%)': return_percent})
            running_total_return += return_percent
        except Exception as e:
            print(f"On date; {buy_date} -encountered error: {e}")



    if results:
        results.append({'Buy Date': '', 'Sell Date': '', 'Return (%)': running_total_return})
    results_df = pd.DataFrame(results)

    #output_fname = f"breakout_info_{gen_ticker_timestamp_suffix_str(ticker)}"
    #results.to_csv(output_fname)
    return results_df


# --- Breakout Detection ---
def detect_breakouts(data, volume_threshold, price_change_threshold):
    # messy 1 line version
    # #breakout_condition = (data[('Volume', ticker)] > 2 * data[('Volume', ticker)].rolling(20).mean()) & (data[('Close', ticker)] > data[('Close', ticker)].shift(1) * 1.02)
    #
    #data["Date"] = data.index
    data['Volume'] = data['Volume'].fillna(method='ffill')
    data['AvgVolume20d'] = data['Volume'].rolling(window=20, min_periods=20, center=False).mean()
    data['AvgVolume20d'] = data['AvgVolume20d'].fillna(method='ffill')

    def calc_volume_change(given_row):
        avg_volume_20d = given_row['AvgVolume20d'].iloc[0] if isinstance(given_row['AvgVolume20d'], pd.Series) else given_row['AvgVolume20d']
        cur_volume     = given_row['Volume'].iloc[0] if isinstance(given_row['Volume'], pd.Series) else given_row['Volume']
        if avg_volume_20d == 0:
            vol_change = 0.0  # avoid division by zero errors
            # potentially change to known flag value for missing or not enough data
        else:
            vol_change = ( cur_volume / avg_volume_20d ) -1
        #print(f"vol change type: {type(vol_change)}")
        #print( vol_change)
        return vol_change

    data['VolumeChange'] = data.apply( calc_volume_change, axis=1)
    #print(data['VolumeChange'])
    #data.to_csv("test.csv")
    data['DailyPrice%Change'] = data['Close'].pct_change(periods=1) * 100

    data["IsDailyPChangeMet"] = data.apply(lambda y: y['DailyPrice%Change'] >= price_change_threshold , axis=1)
    data["IsVolThresholdMet"] = data.apply(lambda x: x['VolumeChange'] >= volume_threshold , axis=1)
    data['IsBreakoutDay'] = data.apply(lambda z: z["IsVolThresholdMet"] & z["IsDailyPChangeMet"] , axis=1 )
    return data



def add_date_column(data:pd.DataFrame)->pd.DataFrame:
    """
    based off of yfinance library that uses date as index by default
    """
    data["Date"] = data.index
    return data


def add_rsi_indicator(data:pd.DataFrame, ticker, rsi_time_period = 14, rsi_oversold_threshold = 30)->pd.DataFrame:
    """
    rsi - relative strength index
     The RSI oscillates between 0 and 100.

     Overbought: When the RSI is above 70, the asset is considered overbought, meaning it may be due for a pullback or reversal.
     Oversold: When the RSI is below 30, the asset is considered oversold, indicating it may be due for a rebound or reversal.
    """
    data['RSI']  = ta.rsi(data[('Close', ticker)], length=rsi_time_period) # relative str index,  RS = avg gain over N / avg loss over N
    #neutral_rsi_val = 50  # currently backfill seems better than adding neutral value
    data['RSI']  = data['RSI'].fillna(method='ffill')
    data["RSI_says_buy"]  = data.apply(lambda y: y['RSI'] < rsi_oversold_threshold , axis=1)
    return data

def add_macd_indicator(data:pd.DataFrame, ticker, macd_fast=12, macd_slow=26, macd_signal=9)->pd.DataFrame:
    """
    # macd   moving avg convergence divergence, MACD = EMA12d - EMA26d    # EMA exponential moving average.
     Crossovers: When the MACD line crosses above the signal line, it’s seen as a bullish signal (indicating potential buying opportunities). Conversely, when the MACD line crosses below the signal line, it’s seen as a bearish signal (indicating potential selling opportunities).
     Divergence: If the price is making new highs but the MACD is not, or vice versa, it can signal a potential reversal or weakening of the current trend.
    """
    macd_result = ta.macd(data[('Close', ticker)], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    #if have_macd_result:

    macd_index  = 0
    macdh_index = 1
    macds_index = 2
    data['MACD']        = macd_result.iloc[:, macd_index]
    data['MACD_hist']   = macd_result.iloc[:, macdh_index]
    data['MACD_signal'] = macd_result.iloc[:, macds_index]
    neutral_macd_val    = 0
    data['MACD']        = data['MACD'].fillna(method='ffill')  #data['MACD'].fillna(neutral_macd_val)
    data['MACD_hist']   = data['MACD_hist'].fillna(method='ffill')  #data['MACD_hist'].fillna(neutral_macd_val)
    data['MACD_signal'] = data['MACD_signal'].fillna(method='ffill')    #data['MACD_signal'].fillna(neutral_macd_val)
    data['MACD_says_buy'] = data.apply(lambda z: z['MACD'] > z['MACD_signal'], axis=1)
    return data


def add_atr_indicator(data:pd.DataFrame, ticker, atr_time_period=14)->pd.DataFrame:
    """
    average true range
     ATR is useful for understanding how much a stock or asset price is fluctuating.
     A higher ATR indicates higher volatility, while a lower ATR indicates lower volatility.
     It doesn't indicate price direction, just the magnitude of price movement.
    """
    data['ATR'] = ta.atr(data[('High', ticker)], data[('Low', ticker)], data[('Close', ticker)], length=atr_time_period)
    data['ATR']  = data['ATR'].fillna(method='ffill')
    data['ATR_says_buy']  = data.apply(lambda w: w['ATR'] > w['ATR'].rolling(window=atr_time_period, min_periods=atr_time_period, center=False).mean(), axis=1)

    # ATR  average true range   ATR
    return data

def add_bollinger_bands(data:pd.DataFrame, ticker, bb_time_period = 20)->pd.DataFrame:
    """
    # bollinger bands Middle Band: 20-period simple moving average (SMA).
    #Upper Band: SMA+2×Standard DeviationSMA+2×Standard Deviation
    #Lower Band: SMA−2×Standard DeviationSMA−2×Standard Deviation


    Reversal Signals:
      Overbought and Oversold Conditions:
        When the price moves to the upper band and starts to reverse, it may signal that the market is overbought and due for a pullback.
        Conversely, when the price moves toward the lower band and begins to bounce, it can signal oversold conditions and a potential buying opportunity.

    Breakouts:
      A breakout above the upper band or below the lower band can signal that a significant price move is occurring, often accompanied by an increase in volume.
      Traders may look to trade the breakout if it’s confirmed by other indicators.

    Band Squeeze:
      If the bands narrow significantly (a squeeze), traders may anticipate a breakout in the direction of the prevailing trend or look for confirmation from other indicators like volume or momentum indicators.

    """
    bbands_result = ta.bbands(data[('Close', ticker)], length=bb_time_period, std=2)
    #have_bbands_result = True if bbands_result is not None else False
    #if have_bbands_result:
    # BBL, BBM, BBU, BBB, BBP
    # bb bollinger band
    # l lower, m middle, u upper, b bounce, p percent
    #print(bbands_result.head)
    bbl_index = 0
    bbm_index = 1
    bbu_index = 2
    bbb_index = 3
    bbp_index = 4
    data['BB_lower']   = bbands_result.iloc[:, bbl_index]
    data['BB_middle']  = bbands_result.iloc[:, bbm_index]
    data['BB_upper']   = bbands_result.iloc[:, bbu_index]
    data['BB_bounce']  = bbands_result.iloc[:, bbb_index]
    data['BB_percent'] = bbands_result.iloc[:, bbp_index]
    data['BB_says_buy']   = data.apply(lambda v: v[('Close', ticker)] > v[('BB_upper', '')], axis=1)
    return data

def add_obv_indicator(data:pd.DataFrame, ticker)->pd.DataFrame:
    """
    obv - on balance volume
     OBV helps traders confirm price trends.
     If the OBV is rising alongside a rising stock price, it suggests that the uptrend is being supported by strong buying volume.
     If OBV is falling while the stock price rises, it can indicate a weakening trend or a potential reversal.
    """
    data['OBV'] = ta.obv(data[('Close', ticker)], data[('Volume', ticker)])
    data['OBV_says_buy']  = data['OBV'] > data['OBV'].shift(1) #data.apply(lambda u: u[('OBV', '')] > u[('OBV', '')].shift(1), axis=1)
    return data



def add_indicators(data, ticker):
    data = add_macd_indicator(data, ticker)
    data = add_rsi_indicator(data, ticker)
    data = add_atr_indicator(data, ticker)
    data = add_bollinger_bands(data, ticker)
    data = add_obv_indicator(data, ticker)
    return data


def add_all_new_indicators(data: pd.DataFrame, ticker) -> pd.DataFrame:
    """
    Adds all new indicators and their combinations to the DataFrame
    """
    #data = add_supertrend_indicator(data, ticker)
    data = add_keltner_channels(data, ticker)
    data = add_vwap_indicator(data, ticker)
    data = add_momentum_indicators(data, ticker)
    data = add_volume_profile(data, ticker)
    #st.write("added all new indicators")
    data = combine_breakout_signals(data, ticker)
    #st.write("combined indicators")
    return data



def validate_ticker(ticker:str)->bool:
    valid_ticker = False
    try:
        ticker_info  = yf.Ticker(ticker).get_info()
        tpr          = 'trailingPegRatio'
        if tpr in ticker_info.keys() and ticker_info[tpr] is not None:
            valid_ticker = True
    except Exception as e:
        print(e)
        # not sure why streamlit has issues with valid tickers here
    return valid_ticker
