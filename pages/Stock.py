import streamlit as st
import pandas as pd
import numpy as np
import math
import IntermarketAnalyzer as ia
from IntermarketAnalyzer import Stock
from IntermarketAnalyzer import K200stock
from IntermarketAnalyzer import R3000stock
from IntermarketAnalyzer import Pair
from IntermarketAnalyzer import Market
from IntermarketAnalyzer import InterMarket
from IntermarketAnalyzer import Portfolio
import datetime
import plotly.graph_objects as go


st.set_page_config(
    page_title="Intermarket Analyzer | Stock",
    page_icon=":bar_chart:",
    # layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
    }
)

stock = None
info = None
gptinfo = None
info_pair = None

tuple_markets = ('KOSPI200', 'RUSSELL3000')

# k200 = Market('KOSPI200')
# r3000 = Market('RUSSELL3000')
# dict_mapping_market = {
#     'KOSPI200': k200,
#     'RUSSELL3000': r3000
# }

dict_mapping_info = {
                'ticker': 'Ticker',
                'NAME': 'Comapny name',
                'description_kor': 'Description',
                'business': 'Business',
                'GPT_comment': 'Comment generated by GPT',
                'GPT_update_info': 'Update info'
            }

st.title('Intermarket Analyzer')
st.header('Stock analysis')
st.write()

selected_marketName = st.selectbox('Select a market', (None,) + tuple_markets[:1], key='target1')
if selected_marketName != None:
    # selected_market = dict_mapping_market[selected_marketName]
    selected_market = Market(selected_marketName)
    tickers = tuple(selected_market.tickers)
    selected_ticker = st.selectbox('Select a stock', tickers, index=7, key='target2')
    if selected_market != None and selected_ticker != None:
        stock = Stock(selected_ticker, selected_marketName)
else:
    st.selectbox('Select market first.', (''), key='target3')
st.divider()

if stock != None:
    df = stock.df
    info = stock.getInfo()
    st.write(f"**{selected_ticker}** / {selected_marketName}")
    st.subheader(f"{info['NAME']}")
    date = df.iloc[-1]['date']
    price = int(df.iloc[-1]['PX_LAST']) if 'KO' in selected_marketName else df.iloc[-1]['PX_LAST']
    price_unit = '원' if 'KO' in selected_marketName else 'USD'
    return_1d = '+'+str(df.iloc[-1]['CHG_PCT_1D'])+'%' if df.iloc[-1]['CHG_PCT_1D'] >= 0 else str(df.iloc[-1]['CHG_PCT_1D'])+'%'
    st.metric(label=f'Price ({date})', value=price, delta=return_1d)

    tab11, tab12, tab13, tab14 = st.tabs(['Overall', 'Dataframe', 'News', 'GPT'])
    with tab11:
        col11, col12 = st.columns(2)
        with col11:
            st.write('전일', int(df.iloc[-2]['PX_LAST']))
            st.write('시가', int(df.iloc[-1]['PX_OPEN']))
            st.write('고가', int(df.iloc[-1]['PX_HIGH']))
            st.write('저가', int(df.iloc[-1]['PX_LOW']))
        with col12:
            st.write('거래량', int(df.iloc[-1]['VOLUME']))
            st.write('시총', math.nan)
            st.write('PER', math.nan)
            st.write('EPS', math.nan)
    with tab12:
        st.dataframe(df)
    with tab14:
        for key, value in stock.getGPTInfo().items():
            subtitle = dict_mapping_info[key]
            st.write(f'<h5>{subtitle}</h5>', unsafe_allow_html=True)
            if key == 'ticker':
                st.code(f"'{value}'")
            elif key == 'business':
                output = ' '.join([f"#{item} " for item in eval(value)]) if value != '[]' else 'GPT의 생성 정보가 없습니다!'
                st.write(output)
            else:
                st.write(value)

        st.divider()
        if st.button('Generate the latest GPT-4 information (paid, 30sec)', key='gpt'):
            with st.spinner('Generating ...'):
                ans = stock.generateGPTanswer(model='gpt-4')
                st.write(ans)
            st.success('Generation complete')

    tab21, tab22, tab23 = st.tabs(['Daily', 'Weekly', 'Monthly'])
    with tab21: 
        stock.plot_candle_streamlit()

    st.subheader(f"Timelag analysis on {info['NAME']}({selected_ticker})")
    selected_marketName_s = st.selectbox('Select source market.', tuple_markets, key='source1')
    if selected_marketName_s == 'KOSPI200':
        df_analysis = pd.read_csv(f"./results-timelag-intermarket/result-timelag-analysis-{selected_marketName}-{selected_marketName_s}-lag0-p10_100-d1000-s5.csv")
    elif selected_marketName_s == 'RUSSELL3000':
        df_analysis = pd.read_csv(f"./results-timelag-intermarket/result-timelag-analysis-{selected_marketName}-{selected_marketName_s}-lag1-p20_100-d1000-s10.csv")
    df_ref = df_analysis[df_analysis['target']==stock.ticker]
    df_ref = df_ref.reset_index(drop=True)
    st.dataframe(df_ref)
    selected_index = st.selectbox('Generate pair: select an index of above DataFrame', range(len(df_ref)))
    row = df_ref.iloc[selected_index]
    st.code(f"pair[{selected_index}] = Pair(Stock('{selected_ticker}', '{selected_marketName}'), Stock('{row['source']}', '{selected_marketName_s}')")

    tab31, tab32 = st.tabs(['Plot', 'Information'])

    with tab31:
        p = ia.getPairFromTimelagAnalysis(df_ref, selected_index, stock)
        lag = df_ref.iloc[0]['lag']
        p.plot_price_streamlit()
        st.divider()   
        st.write(f'<h5>Historical timelag correlation</h5>', unsafe_allow_html=True)      
        p.corrmatrix(lag=lag, min_period=10, max_period=100, max_shift=1000, stride=5)
        p.plot_heatmap_streamlit(lag=lag)
        st.divider()   
        st.write(f'<h5>Periodic timelag correlation</h5>', unsafe_allow_html=True)                    
        p.plot_periodicCorr_streamlit(lag=lag)
        st.divider()   
        st.write(f'<h5>Timelag correlation calendar</h5>', unsafe_allow_html=True)                    
        p.plot_nonZeros_stramlit(lag=lag)
        st.divider()   

    with tab32:
        info_pair = p.getInfo()

        st.subheader('GICS similarity score')
        st.write(info_pair['similarity'])

        col1, col2 = st.columns(2)
        with col1:
            for key, value in info_pair['info_gpt_target'].items():
                subtitle = dict_mapping_info[key] if key != 'ticker' else 'Target ticker'
                st.write(f'<h5>{subtitle}</h5>', unsafe_allow_html=True)
                if key == 'ticker':
                    st.code(f"'{value}'")
                elif key == 'business':
                    output = ' '.join([f"#{item} " for item in eval(value)]) if value != '[]' else 'GPT의 생성 정보가 없습니다!'
                    st.write(output)
                else:
                    st.write(value)
                    
        with col2:
            for key, value in info_pair['info_gpt_source'].items():
                subtitle = dict_mapping_info[key] if key != 'ticker' else 'Source ticker'
                st.write(f'<h5>{subtitle}</h5>', unsafe_allow_html=True)
                if key == 'ticker':
                    st.code(f"'{value}'")
                elif key == 'business':
                    output = ' '.join([f"#{item} " for item in eval(value)]) if value != '[]' else 'GPT의 생성 정보가 없습니다!'
                    st.write(output)
                else:
                    st.write(value)

    with st.form(key=f'settning_detail'):
            date2 = st.date_input(' Set date', datetime.date(2023, 6, 1))
            period2 = st.slider(' Set period', 5, 300, 60)
            lag2_default = 0 if selected_marketName_s == 'KOSPI200' else 1
            lag2 = st.number_input(' Set time lag',value=lag2_default, min_value=0, max_value=100, step=1)
            st.write('Selected','(', 'date =', date2, 'period = ', period2, 'lag = ', lag2, ')')
            submit2 = st.form_submit_button('APPLY')
    
    p2 = p.set_date(date2).set_period(period2).set_lag(lag2)
    p2.plot_compare_streamlit()
    p2.plot_returns_streamlit()