from datetime      import datetime, date, timedelta

import streamlit as st
import pandas as pd


from yfinance_utils import (
    add_all_new_indicators,
    add_date_column,
    add_indicators,
    calculate_returns,
    detect_breakouts,
    fetch_data,
    validate_ticker
)

st.title("Stock Breakout Strategy Backtester")

if 'date_state' not in st.session_state:
    st.session_state.date_state = {
        'start_date': pd.to_datetime('2020-01-01'),
        'end_date': pd.to_datetime('2024-01-01'),
        'min_date': pd.to_datetime('1985-01-01'),  # yahoo finance doesn't support queries past 1970, get wonkiness if go before stock exists- needs more error handling
        'max_date': pd.to_datetime(date.today())  # Today's date
    }



ticker = st.text_input("Ticker Symbol (e.g., AAPL)", "AAPL").upper()
# would be nice to have robust error handling on invalid symbol and dates
col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input(
        "Start Date",
        value=st.session_state.date_state['start_date'],
        min_value=st.session_state.date_state['min_date'],
        max_value=st.session_state.date_state['max_date']-timedelta(days=1)
    )



with col2:

    end_date = st.date_input(
        "End Date",
        value=st.session_state.date_state['end_date'],
        min_value=st.session_state.date_state['min_date']+timedelta(days=1),
        max_value=st.session_state.date_state['max_date']
    )


volume_threshold = st.number_input("Volume Breakout Threshold (e.g., 2.0 for 200%)", value=2.0)
price_change_threshold = st.number_input("Daily Price Change Threshold (e.g., 2.0 for 2%)", value=2.0)
holding_period = st.number_input("Holding Period (days)", value=10)


if 'params' not in st.session_state:
    st.session_state.params = {
        'ticker': None,
        'start_date': None,
        'end_date': None,
        'volume_threshold': None,
        'price_change_threshold': None,
        'holding_period': None
    }


params_changed = (
    st.session_state.params['ticker'] != ticker or
    st.session_state.params['start_date'] != start_date or
    st.session_state.params['end_date'] != end_date or
    st.session_state.params['volume_threshold'] != volume_threshold or
    st.session_state.params['price_change_threshold'] != price_change_threshold or
    st.session_state.params['holding_period'] != holding_period
)

if st.button("Generate Report"): # and (params_changed and 'data' in st.session_state):
# if st.button("Generate Report"):
    st.session_state.report_generated = False

    if not validate_ticker(ticker):
        st.error(f"Was unable to find given ticker {ticker} on yahoo finance")
        st.stop()

    # in theory should not be feasible if using ui, but can manually type dates
    # which streamlit has KNOWN bugs that for w/e reason they relabel as feature enhancements X_X
    if start_date > end_date:
        temp_end_date = end_date
        end_date = start_date
        start_date = temp_end_date

    max_period = (end_date - start_date).days
    min_period = 2
    if holding_period < min_period:
        st.error(f"Holding period was found to be {holding_period} days, program requires a minimum of {min_period} days")
        st.stop()
    if holding_period > max_period:
        st.error(f"Holding period was found to be {holding_period} days, which is greater than date range selected of {start_date} to {end_date}")
        st.stop()
    min_date_window = 50 # anything below this tends to result in indicators going wonky
    # seeminly more logical to require min date window vs trying to account for incredibly small window
    if max_period < min_date_window:
        st.error(f"Date range was found to be {start_date} to {end_date}, program requires a minumum window of {min_date_window}days to function properly")
        st.stop()

    progress_bar = st.progress(0, text="Operation in progress. Please wait.")
    st.session_state.ticker = ticker
    st.session_state.params['start_date'] = start_date
    st.session_state.params['end_date'] = end_date
    st.session_state.volume_threshold = volume_threshold
    st.session_state.price_change_threshold = price_change_threshold
    st.session_state.holding_period = holding_period


    try:
        st.session_state.data = fetch_data(ticker, start_date, end_date)
        progress_bar.progress(5, text=f"Data for {ticker} obtained")

        st.session_state.data = add_date_column(st.session_state.data)
        progress_bar.progress(6, text=f"Date column explicitly added for {ticker}")

        st.session_state.data = detect_breakouts(st.session_state.data, volume_threshold, price_change_threshold)
        progress_bar.progress(16, text=f"Original Breakout strategy added")

        st.session_state.data = add_indicators(st.session_state.data, ticker)
        progress_bar.progress(26, text=f"Initial set of expanded indicators added")

        st.session_state.data = add_all_new_indicators(st.session_state.data, ticker)
        progress_bar.progress(36, text=f"Secondary set of expanded indicators added")

        #st.session_state.data = (st.session_state.data, ticker)
        progress_bar.progress(40, text="Indicators combined into subcategories for buy indicators to test")





        #Original Breakout
        st.session_state.breakout_data = st.session_state.data[st.session_state.data['IsBreakoutDay']]
        st.session_state.breakout_days = st.session_state.breakout_data['Date'].to_list()
        st.session_state.original_results = calculate_returns(st.session_state.data, ticker, st.session_state.breakout_days, holding_period)
        progress_bar.progress(50, text = "original break out calculated")

        #Momentum Volume Breakout
        st.session_state.breakout_mvb       = st.session_state.data[st.session_state.data['Momentum_Volume_Breakout']]
        st.session_state.breakout_days_mvb  = st.session_state.breakout_mvb['Date'].to_list()
        st.session_state.mvb_results        = calculate_returns(st.session_state.data, ticker, st.session_state.breakout_days_mvb, holding_period)
        progress_bar.progress(60, text = "momentum volume break out calculated")

        #Trend Confirmation Breakout
        st.session_state.breakout_tcb       = st.session_state.data[st.session_state.data['Trend_Confirmation_Breakout']]
        st.session_state.breakout_days_tcb  = st.session_state.breakout_tcb['Date'].to_list()
        st.session_state.tcb_results        = calculate_returns(st.session_state.data, ticker, st.session_state.breakout_days_tcb, holding_period)
        progress_bar.progress(70, text = "trend confirmation break out calculated")


        #Volatility Breakout
        st.session_state.breakout_vb        = st.session_state.data[st.session_state.data['Volatility_Breakout']]
        st.session_state.breakout_days_vb   = st.session_state.breakout_vb['Date'].to_list()
        st.session_state.vb_results         = calculate_returns(st.session_state.data, ticker, st.session_state.breakout_days_vb, holding_period)
        progress_bar.progress(80, text = "volatility break out calculated")


        #Volume Price Breakout
        st.session_state.breakout_vpb       = st.session_state.data[st.session_state.data['Volume_Price_Breakout']]
        st.session_state.breakout_days_vpb  = st.session_state.breakout_vpb['Date'].to_list()
        st.session_state.vpb_results        = calculate_returns(st.session_state.data, ticker, st.session_state.breakout_days_vpb, holding_period)
        progress_bar.progress(90, text = "volume price break out calculated")


        #Ultimate Breakout
        st.session_state.breakout_ub        = st.session_state.data[st.session_state.data['Ultimate_Breakout']]
        st.session_state.breakout_days_ub   = st.session_state.breakout_ub['Date'].to_list()
        st.session_state.ub_results         = calculate_returns(st.session_state.data, ticker, st.session_state.breakout_days_ub, holding_period)
        progress_bar.progress(100, text = "ultimate break out calculated")




        st.session_state.report_generated = True

    except Exception as e:
        st.error(f"An error occurred: {e}")



if 'report_generated' in st.session_state and st.session_state.report_generated:
    st.subheader("Original Breakout Strategy Results")
    st.dataframe(st.session_state.original_results)
    if st.session_state.original_results.empty:
        st.write("No results found using original breakout strategy")

    st.subheader("Momentum Volume Breakout Strategy Results")
    st.dataframe(st.session_state.mvb_results)
    if st.session_state.mvb_results.empty:
        st.write("No results found using Momentum Volume breakout strategy")

    st.subheader("Trend Confirmation Breakout Strategy Results")
    st.dataframe(st.session_state.tcb_results)
    if st.session_state.tcb_results.empty:
        st.write("No results found using Trend Confirmation breakout strategy")

    st.subheader("Volatility Breakout Strategy Results")
    st.dataframe(st.session_state.vb_results)
    if st.session_state.vb_results.empty:
        st.write("No results found using Volatility breakout strategy")

    st.subheader("Volume Price Breakout Strategy Results")
    st.dataframe(st.session_state.vpb_results)
    if st.session_state.vpb_results.empty:
        st.write("No results found using Volume Price breakout strategy")

    st.subheader("Pseudo Ultimate Breakout Strategy Results")
    st.dataframe(st.session_state.ub_results)
    if st.session_state.ub_results.empty:
        st.write("No results found using Pseudo Ultimate breakout Strategy Results")

    combined_results = pd.concat(
        [
            st.session_state.original_results,
            st.session_state.mvb_results,
            st.session_state.tcb_results,
            st.session_state.vb_results,
            st.session_state.vpb_results,
            st.session_state.ub_results
        ],

        keys=['Original', 'Momentum_Volume', 'Trend_Confirmation', 'Volatility', 'Volume_Price', 'Pseudo_Ultimate']
    )
    st.session_state.csv = combined_results.to_csv(index=True)


    st.download_button(
        label="Download data as CSV",
        data=st.session_state.csv,
        file_name=f"breakout_results_{gen_ticker_timestamp_suffix_str(ticker)}.csv",
        mime='text/csv',
    )

