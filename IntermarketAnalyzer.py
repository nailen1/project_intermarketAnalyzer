
### version.230717

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr 
from scipy.stats.mstats import winsorize
from tslearn.metrics import dtw
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import random
import time
from datetime import datetime
import math
import os
import re
import ast
import openai
import multiprocessing as mp
from joblib import Parallel, delayed
from tqdm import tqdm
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.subplots as sp

from plotly_calplot import calplot

class Stock:
    def __init__(self, ticker, marketName):
        self.ticker = ticker
        self.marketName = marketName
        self.df = self.df()
        self.info = None
        
    def df(self):
        file_ticker = self.ticker.replace('/', '_') if '/' in self.ticker else self.ticker 
        folder_path = f'./datasets-{self.marketName}/'
        df = pd.read_csv(folder_path+ f'dataset-{self.marketName}-{file_ticker}.csv') 
        return df

    def plot_candle_streamlit(self):
        df= self.df
            
        candlestick = go.Candlestick(x=df['date'],
                                    open=df['PX_OPEN'],
                                    high=df['PX_HIGH'],
                                    low=df['PX_LOW'],
                                    close=df['PX_LAST'],
                                    increasing_line_color='red',  # 양봉일 때 색상
                                    decreasing_line_color='blue',  # 음봉일 때 색상
                                    name='Candlestick')

        # 거래량 막대 차트 데이터
        volume = go.Bar(x=df['date'], y=df['VOLUME'], name='Volume')

        # 서브플롯 생성
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.2)

        # 캔들 차트 추가
        fig.add_trace(candlestick, row=1, col=1)

        # 거래량 막대 차트 추가
        fig.add_trace(volume, row=2, col=1)

        # 캔들 차트 상하 길이 조정
        fig.update_layout(yaxis=dict(domain=[0.4, 1]))

        # y축 단위 설정
        fig.update_yaxes(title_text='Price', row=1, col=1)
        fig.update_yaxes(title_text='Volume', row=2, col=1)
        fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))

        # 그래프 출력
        st.plotly_chart(fig)

    def getInfo(self):
        folder_path = f'./datasets-info/'
        file_name = f'dataset-{self.marketName}-info.csv'
        if os.path.exists(folder_path + file_name):
            df = pd.read_csv(folder_path + file_name) 
            info = df[df['ticker']==self.ticker].to_dict(orient='records')[0]            
        else:
            info = { 'ticker': self.ticker }
        self.info = info
        return self.info

    def getGPTInfo(self):
        folder_path = f'./datasets-GPTanswer/'
        file_name = f'dataset-{self.marketName}-gpt.csv'
        if os.path.exists(folder_path + file_name):
            df = pd.read_csv(folder_path + file_name) 
            info_gpt = df[df['ticker']==self.ticker].to_dict(orient='records')[0]            
        else:
            info_gpt = { 'ticker': self.ticker }
        self.info_gpt = info_gpt
        return self.info_gpt

    def generateGPTanswer(self, model = 'gpt-3.5-turbo', lang='ko'):
        self.getInfo()
        info = self.info

        LAM_GPT_API_KEY = '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'
        openai.api_key = LAM_GPT_API_KEY
        model = model

        if lang == 'ko':
            c = [
                '넌 전문적인 기업분석가 또는 투자 애널리스트 역할을 맡아줘.', 
                f'''
                자, 이제 내가 너에게 한 기업의 정보를 다음과 같이 알려줄게: 
                기업 이름(name): {info['NAME']}
                티커(ticker): {info['ticker']}
                요약 정보(description): {info['CIE_DES']}

                이제 위 기업에 대해, 다음과 같은 키와 값을 가진 파이썬 딕셔너리를 코드블록 형태로 답변을 적어줘.

                'description_kor': 내가 알려준 기업의 요약 정보(description)의 자연스러운 한글 번역 결과
                'business': 내가 알려준 기업의 사업 내용을 항목화한 파이썬 리스트 
                'gpt_comment': 네가 생각하는 이 기업의 장단점을 간략하게 요약해서 자연스러운 존칭 표현의 문장 형태로 적어줘. 

                각 항목에 대해 정보가 없거나 모르겠으면 값을 빈칸으로 비워줘. 네가 아는 한 최선의 정보를 부탁해!
                '''
                ]
        elif lang == 'en':
            c = [
                'Please take on the role of a professional business analyst or investment analyst.', 
                f'''
                Alright, now I'm going to provide you with information about a company as follows:
                Company Name: {info['NAME']}
                Ticker: {info['ticker']}
                Summary Information: {info['CIE_DES']}

                Now, for the given company, please provide a Python dictionary with the following keys and values, in code block format:
                'business': A Python list itemizing the business activities of the mentioned company.
                'gpt_comment': Please provide a brief and natural summary of the strengths and weaknesses of this company from your perspective.

                If you don't have information or are unsure about any of the items, please leave the values blank. I trust that you'll provide the best information you have!
                '''
                ]        

        q = openai.ChatCompletion.create(
            model = model,
            messages = [
                {
                    "role": "system", 
                    "content": c[0]
                },
                {
                    "role": "user", 
                    "content": c[1]
                }
            ]
        )

        a = q.choices[0].message.content
        print(a)
        index_prohibit = a.find("print(")
        if index_prohibit != -1:
            a = a[:index_prohibit]

        list_pattern = r"\{.*\}"
        if re.search(list_pattern, a, re.DOTALL) == None:
            dct = info
        else:
            str_dict = re.findall(list_pattern, a, re.DOTALL)[0]
            dct = ast.literal_eval(str_dict)

        info_gpt = {}
        info_gpt['ticker'] = info['ticker']
        info_gpt['NAME'] = info['NAME']

        if lang == 'ko':
            info_gpt['description_kor'] = dct['description_kor'] 
            info_gpt['business'] = dct['business'] 
            if dct['gpt_comment'] != '':
                gpt_comment = dct['gpt_comment'] + " 이 코멘트는 웹 상의 데이터로 학습한 거대한 언어 모델의 출력 결과입니다. 답변의 정확성을 보장할 수는 없습니다." 
            info_gpt['GPT_comment'] = gpt_comment

            current_datetime = datetime.now().strftime("%Y-%m-%d")
            info_gpt['GPT_update_info'] = f'powered by {model} at {current_datetime}'
        
        elif lang == 'en':
            info_gpt['description'] = info['CIE_DES']
            info_gpt['business'] = dct['business'] 
            if dct['gpt_comment'] != '':
                gpt_comment = dct['gpt_comment'] + " Please note that this comment is the output of a Large Language Model trained on web data. The accuracy of the response cannot be guaranteed."
            info_gpt['GPT_comment'] = gpt_comment

            current_datetime = datetime.now().strftime("%Y-%m-%d")
            info_gpt['GPT_update_info'] = f'powered by {model} at {current_datetime}'

        self.info_gpt = info_gpt

        return self.info_gpt


class K200stock(Stock):
    def __init__(self, ticker):
        super().__init__(ticker, marketName='KOSPI200')

class R3000stock(Stock):
    def __init__(self, ticker):
        super().__init__(ticker, marketName='RUSSELL3000')

class Pair:
    def __init__(self, Stock_target, Stock_source):
        self.target = Stock_target
        self.source = Stock_source 
        self.df = self.df()
        self.period = 75 # default: period = 75
        self.lag = 0
        self.lead = 0
        self.shift = 0
        self.date = None
        self.date_initial = None
        self.returns_t = None
        self.returns_s = None
        self.date_target = self.df.iloc[-1]['date']
        self.date_source = self.date_target
        self.df_result = pd.DataFrame()
        self.price_compare = pd.DataFrame()
        self.df_roll = pd.DataFrame()
        self.corr_matrix = pd.DataFrame()
        self.corr_mean = pd.DataFrame()
        self.params = self.record_params()
        self.corr_stat2 = None
        self.apply()
        
    def df(self):
        # if isinstance(self.target, Stock) and isinstance(self.source, Stock):
        #     df_t = self.target.df[['date', 'CHG_PCT_1D', 'PX_LAST', 'VOLUME', 'MOV_AVG_5D']]
        #     df_s = self.source.df[['date', 'CHG_PCT_1D', 'PX_LAST', 'VOLUME', 'MOV_AVG_5D']]
        # elif isinstance(self.target, Portfolio) or isinstance(self.source, Portfolio):
        #     df_t = self.target.df[['date', 'CHG_PCT_1D']]
        #     df_s = self.source.df[['date', 'CHG_PCT_1D']]

        df_t = self.target.df[['date', 'CHG_PCT_1D', 'PX_LAST', 'VOLUME', 'MOV_AVG_5D']]
        df_s = self.source.df[['date', 'CHG_PCT_1D', 'PX_LAST', 'VOLUME', 'MOV_AVG_5D']]

        df_merge = df_t.merge(df_s, on='date', how='outer', suffixes=('_t', '_s'))
        df_merge = df_merge.sort_values('date')
        df_merge[['CHG_PCT_1D_t', 'CHG_PCT_1D_s']] = df_merge[['CHG_PCT_1D_t', 'CHG_PCT_1D_s']].fillna(0)
        df_merge = df_merge.fillna(method='ffill')
        df = df_merge.reset_index(drop=True)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def getInfo(self):
        info = {
            'target': self.target.ticker, 'source': self.source.ticker, 
            'info_target': self.target.getInfo(), 'info_source': self.source.getInfo(),
            'info_gpt_target': self.target.getGPTInfo(), 'info_gpt_source': self.source.getGPTInfo(),
            }
        self.info = info
        self.info['similarity'] = self.getSimilarity()
        return self.info

    def getSimilarity(self, option=None):
        info_target = self.info['info_target']
        info_source = self.info['info_source']
        del info_target['NAME']
        del info_target['CIE_DES']
        del info_source['NAME']
        del info_source['CIE_DES']
            
        total_items = len(info_target)
        matching_items = sum(info_target.get(key) == info_source.get(key) for key in info_target)
        similarity_score = (matching_items / total_items) * 10

        # if option == 'gpt':
            # prompting ...
        
        return similarity_score

    def set_date(self, date=None):
        self.date = pd.to_datetime(date)
        self.record_params()                                        
        return self.apply()
    
    def set_date_initial(self, date_initial=None):
        self.date_initial = pd.to_datetime(date_initial)
        self.record_params()
        return self.apply()

    def set_period(self, period=75):
        self.period = period
        self.record_params()                                
        return self.apply()
    
    def set_lag(self, lag=0):
        self.lag = lag
        self.record_params()                        
        return self.apply()

    def set_lead(self, lead=0):
        self.lead = lead
        self.record_params()                
        return self.apply()
    
    def set_shift(self, shift):
        self.shift = shift
        self.record_params()        
        return self.apply()
    
    def set_params(self, date, period, lag, lead, shift):
        self.date = date
        self.period = period
        self.lag = lag
        self.lead = lead
        self.shift = shift
        self.record_params()
        return self.apply()
    
    def reset(self):
        self.period = 75 # default: period = 75
        self.lag = 0
        self.lead = 0
        self.shift = 0
        self.date = None
        self.returns_t = None
        self.returns_s = None
        self.date_target = self.df.iloc[-1]['date']
        self.date_source = self.date_target
        self.df_result = pd.DataFrame()
        self.params = self.record_params()
        return self.apply()

    def record_params(self):
        self.params = {'date': self.date, 'period': self.period, 'lag': self.lag, 'lead': self.lead, 'shift': self.shift, 'date_target': self.date_target, 'date_source': self.date_source }
        return self.params
    
    def record_time(self):
        now = datetime.now()
        formatted_date = now.strftime("%Y%m%d%H")
        return formatted_date

    def apply(self):    
        df = self.df        
        if self.date != None:
            df_ref = df[df['date'] <= self.date]
        else:
            df_ref = df 
            if len(df_ref) < self.period:
                raise ValueError("(parameter error) date is too close.")
        
        # if self.date_initial != None:
        #     self.period = 0
        #     self.shift = 0
        #     df_ref = df_ref[df_ref['date'] >= self.date_initial]

        index_ref_f = df_ref.index[-1]
        index_ref_i = index_ref_f - self.period
       
        index_target_i = index_ref_i + self.shift + self.lead 
        index_source_i = index_ref_i + self.shift - self.lag

        index_target_f = index_target_i + self.period
        index_source_f = index_source_i + self.period

        if index_target_f > df.index[-1]:
            index_target_f = None

        self.data_target = self.df[['date', 'CHG_PCT_1D_t', 'PX_LAST_t', 'VOLUME_t', 'MOV_AVG_5D_t']].iloc[index_target_i: index_target_f]
        if index_target_f == -1:
            index_source_f = index_source_i + len(self.data_target)
        self.data_source = self.df[['date', 'CHG_PCT_1D_s', 'PX_LAST_s', 'VOLUME_s', 'MOV_AVG_5D_s']].iloc[index_source_i: index_source_f]        
        self.date_target = self.data_target.iloc[-1]['date']
        self.date_source = self.data_source.iloc[-1]['date']
        self.record_params()
        return self

    def corr(self, winsor=True, limit=0.05):
        arr_target = np.array(self.data_target['CHG_PCT_1D_t'])
        arr_source = np.array(self.data_source['CHG_PCT_1D_s'])

        if winsor == True:
            arr_target = winsorize(arr_target, limits=[limit, limit])
            arr_source = winsorize(arr_source, limits=[limit, limit])
                    
        self._corr, self.p_value = pearsonr(arr_target, arr_source)
        self.dtw()
        return {'target': self.target.ticker, 'source': self.source.ticker, 'lag': self.lag, 'shift': self.shift, 'corr': self._corr, 'dtw': self._dtw, 'p_value': self.p_value, 'date_target': self.date_target, 'date_source': self.date_source, 'period': self.period}

    def corr_price(self, winsor=True, limit=0.05):
        arr_target = np.array(self.data_target['PX_LAST_t'])
        arr_source = np.array(self.data_source['PX_LAST_s'])

        if winsor == True:
            arr_target = winsorize(arr_target, limits=[limit, limit])
            arr_source = winsorize(arr_source, limits=[limit, limit])
                    
        self._corr, self.p_value = pearsonr(arr_target, arr_source)
        return {'target': self.target.ticker, 'source': self.source.ticker, 'lag': self.lag, 'shift': self.shift, 'corr': self._corr, 'p_value': self.p_value, 'date_target': self.date_target, 'date_source': self.date_source, 'period': self.period}

    def corr_vol(self, winsor=True, limit=0.05):
        arr_target = np.array(self.data_target['VOLUME_t'])
        arr_source = np.array(self.data_source['VOLUME_s'])

        if winsor == True:
            arr_target = winsorize(arr_target, limits=[limit, limit])
            arr_source = winsorize(arr_source, limits=[limit, limit])
                    
        self._corr, self.p_value = pearsonr(arr_target, arr_source)
        return {'target': self.target.ticker, 'source': self.source.ticker, 'lag': self.lag, 'shift': self.shift, 'corr': self._corr, 'p_value': self.p_value, 'date_target': self.date_target, 'date_source': self.date_source, 'period': self.period}

    def corr_ma5(self, winsor=True, limit=0.05):
        arr_target = np.array(self.data_target['MOV_AVG_5D_t'])
        arr_source = np.array(self.data_source['MOV_AVG_5D_s'])

        if winsor == True:
            arr_target = winsorize(arr_target, limits=[limit, limit])
            arr_source = winsorize(arr_source, limits=[limit, limit])
                    
        self._corr, self.p_value = pearsonr(arr_target, arr_source)
        return {'target': self.target.ticker, 'source': self.source.ticker, 'lag': self.lag, 'shift': self.shift, 'corr': self._corr, 'p_value': self.p_value, 'date_target': self.date_target, 'date_source': self.date_source, 'period': self.period}

    def fast_corr(self, winsor=True, limit=0.05):
        arr_target = np.array(self.data_target['CHG_PCT_1D_t'])
        arr_source = np.array(self.data_source['CHG_PCT_1D_s'])

        if winsor == True:
            arr_target = winsorize(arr_target, limits=[limit, limit])
            arr_source = winsorize(arr_source, limits=[limit, limit])
                    
        corr = np.corrcoef(arr_target, arr_source)[0,1]
        return corr

    def result(self, winsor=True, limit=0.05, solver='return'):
        if solver == 'return':
            self.solve = self.corr
        elif solver == 'price':
            self.solve = self.corr_price
        elif solver == 'volume':
            self.solve = self.corr_vol
        elif solver == 'ma5':
            self.solve = self.corr_ma5
        elif solver == 'dtw':
            self.solve = self.dtw

        start_time = time.time()
        bucket = []
        for i in range(self.period):
            dictSol = self.set_lag(i).solve(winsor=winsor, limit=limit)
            bucket.append(dictSol)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"end: {execution_time} seconds elapsed")
        self.df_result = pd.DataFrame(bucket)
        self.set_lag(0)

        # def f(i):
        #     return self.set_lag(i).solve(winsor=winsor, limit=limit)

        # start_time = time.time()
        # my_core = mp.cpu_count()
        # with Parallel(n_jobs=my_core) as parallel:
        #     results = parallel(delayed(f)(i) for i in range(self.period))
        # end_time = time.time()
        # execution_time = end_time - start_time
        # print(f"end: {execution_time} seconds elapsed")
        # self.df_result = pd.DataFrame(results)
        # self.set_lag(0)

        return self.df_result
    
    def solution(self, rank=1, key=abs, winsor=True, limit=0.05, solver='return'):
        if solver == 'return':
            self.solve = self.corr
        elif solver == 'price':
            self.solve = self.corr_price
        elif solver == 'volume':
            self.solve = self.corr_vol
        elif solver == 'ma5':
            self.solve = self.corr_ma5
        elif solver == 'dtw':
            self.solve = self.dtw

        if rank == 1:
            bucket = []
            for i in range(self.period):
                dictSol = self.set_lag(i).solve(winsor=winsor, limit=limit)
                bucket.append(dictSol)
            rank_info = max(bucket, key=lambda x: abs(x['corr']))
            # 효율 차원 절대값 선택 옵션 적용 안함
        else:
            self.result(winsor=winsor, limit=limit, solver=solver)
            sorted_df = self.df_result.sort_values(by='corr', key=key, ascending=False)
            rank_info = sorted_df.iloc[rank-1].to_dict()        
        
        self.set_lag(0)
        return rank_info

    def dtw(self):
        initial_price_target = self.data_target.iloc[0]['PX_LAST_t']
        initial_price_source = self.data_source.iloc[0]['PX_LAST_s']
        if math.isnan(initial_price_target) or math.isnan(initial_price_source):
            self._dtw = np.nan
            return self._dtw
    
        arr_target = np.array(self.data_target['PX_LAST_t']/initial_price_target)
        arr_source = np.array(self.data_source['PX_LAST_s']/initial_price_source)

        self._dtw = dtw(arr_target, arr_source)
        return self._dtw

    def result_dtw(self, winsor=True, limit=0.05):
        bucket = []
        for i in range(self.period):
            dictSol = self.set_lag(i).dtw()
            bucket.append(dictSol)
        self.df_result_dtw = pd.DataFrame(bucket)
        self.set_lag(0)
        return self.df_result_dtw
    
    def solution_dtw(self, rank=1, key=abs, winsor=True, limit=0.05):
        if rank == 1:
            bucket = []
            for i in range(self.period):
                dictSol = self.set_lag(i).dtw()
                bucket.append(dictSol)
            rank_info = min(bucket, key=lambda x: abs(x['dtw']))
            # 효율 차원 절대값 선택 옵션 적용 안함
        else:
            self.result_dtw()
            sorted_df = self.df_result_dtw.sort_values(by='dtw', key=key, ascending=True)
            rank_info = sorted_df.iloc[rank-1].to_dict()        
        self.set_lag(0)
        return rank_info

    def corrmatrix(self, lag, min_period=10, max_period=100, max_shift=250*4, stride=5, winsor=True, limit=0.05, plot=True, save=True, folder_name=None):
        # max_shift = historicity
        print('start analysis: correlation heatmap ...')
        print(f'lag={lag}: source({self.source.ticker}) -> target({self.target.ticker}) ...')
        
        start_time = time.time()
        df = self.df
        df = df if self.date == None else df[df['date']<=self.date] 

        config_max_shift = max_shift
        if max_shift > len(df):
            max_shift = len(df) - 10
            
        arr_target = np.array(df['CHG_PCT_1D_t'])
        arr_source = np.array(df['CHG_PCT_1D_s'])
        arr_date = np.array(df[df['date']<=self.date_target].iloc[-max_shift:]['date'])

        bucket = []
        if winsor == True:
            arr_target = winsorize(arr_target, limits=[limit, limit])
            arr_source = winsorize(arr_source, limits=[limit, limit])

        for period in range(min_period, max_period+1, stride):
            bucket_by_period = []
            for shift in range(max_shift, 0, -1):
                target = arr_target[-(period+shift):-(shift)]
                source = arr_source[-(period+shift+lag):-(shift+lag)]
                try:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        correlation = np.corrcoef(target, source)[0, 1]
                except Exception as err:
                    correlation = 0

                n = len(target)
                with np.errstate(divide='ignore', invalid='ignore'):
                    t = correlation * np.sqrt((n - 2) / (1 - correlation**2))
                    p_value = 2 * (1 - stats.t.cdf(abs(t), df=n-2))

                corr = correlation if p_value < 0.05 else 0
                bucket_by_period.append(corr)

                # corr, p_value = pearsonr(t,s)
                # corr = corr if p_value < 0.05 else 0
                # bucket_by_period.append(corr)
            bucket.append(bucket_by_period)

        corr_matrix = pd.DataFrame(bucket)
        corr_matrix.columns = arr_date
        corr_matrix.index = range(min_period, max_period+1, stride)

        self.corr_matrix = corr_matrix

        end_time = time.time()
        elapsed_time = end_time - start_time

        print('end ...') 
        print('total elapsed time: %.2f seconds' %elapsed_time)
        print()

        if plot == True:
            self.plot_heatmap(lag)

        if save == True:
            target_ticker = self.target.ticker.replace('/', '_')
            source_ticker = self.source.ticker.replace('/', '_')
            folder_name_1 = f"results-corrmatrix-{self.record_time()}" if folder_name==None else folder_name
            folder_name_2 = f"results-corrmatrix-{target_ticker}-pairs-lag{lag}-p{max_period}-d{config_max_shift}-s{stride}"
            print(folder_name_1)
            print(folder_name_2)

            path_1 = f'./{folder_name_1}'
            path_2 = f'./{folder_name_1}/{folder_name_2}'

            if not os.path.exists(path_1):
                os.makedirs(path_1)
            if not os.path.exists(path_2):
                os.makedirs(path_2)

            file_name = f'corrmatrix-lag{lag}-{target_ticker}-{source_ticker}'
            corr_matrix.to_csv(f'{path_2}/{file_name}.csv', index_label='period')

        return self.corr_matrix

    def openCorrmatrixFile(self, lag, folder_subname, min_period=20, max_period=100, max_shift=1000, stride=5):
        ticker_target = self.target.ticker.replace('/', '_')
        ticker_source = self.source.ticker.replace('/', '_')
        folder_name = f'results-corrmatrix-{folder_subname}'
        subfolder_name=f'results-corrmatrix-{self.target.ticker}-pairs-lag{lag}-p{max_period}-d{max_shift}-s{stride}'
        folder_path = f'./{folder_name}/{subfolder_name}/'
        file_name = folder_path+f'corrmatrix-lag{lag}-{ticker_target}-{ticker_source}.csv'
        df = openCSV(file_name)    

        self.corr_matrix = df
        return self.corr_matrix

    def plot_heatmap(self, lag):
        if self.corr_matrix.empty:
            raise EmptyError("Execute pair.corrmatrix() first.")
        corr_matrix = self.corr_matrix
        plt.figure(figsize=(30, 10))  # 그래프의 크기 설정
        ax = sns.heatmap(corr_matrix, vmin=-1, vmax=1, cmap='coolwarm', annot=False, fmt=".2f")
        ax.set_title(f'Correlation heatmap: lagged {lag} source({self.source.ticker}) -> target({self.target.ticker}) ...')
        ax.set_xlabel('date')
        ax.set_ylabel('period')
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        plt.show()

    def plot_heatmap_streamlit(self, lag):
        df = self.corr_matrix
        fig = go.Figure(data=go.Heatmap(
            z=df.values,
            x=df.columns,
            y=df.index,
            colorscale='RdBu_r',
            zmin=-1,
            zmax=1,
            hovertemplate='Column: %{x}<br>Index: %{z:.2f}<extra></extra>'
        ))
        fig.update_traces(colorbar_orientation='h', selector=dict(type='heatmap')) 
        fig.update_traces(colorbar_ticklabelposition='inside bottom', selector=dict(type='heatmap'))
        fig.update_layout(shapes=[go.layout.Shape(
            type='rect',
            xref='paper',
            yref='paper',
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line={'width': 1, 'color': 'black'}
        )])
        st.plotly_chart(fig)

    def getCorrMean(self):
        if self.corr_matrix.empty:
            raise EmptyError("Execute pair.corrmatrix() first.")
        corr_matrix = self.corr_matrix
        df = corr_matrix.mean().to_frame('corr_mean')
        df['corr_mean_MA(20)'] = df['corr_mean'].rolling(window=20, min_periods=1).mean()
        df.index = pd.to_datetime(df.index)
        self.corr_mean = df
        return df

    def getCorrStat(self):
        if self.corr_mean.empty:
            df = self.getCorrMean()
        else:
            df = self.corr_mean
        corr_stat = df.describe()
        self.corr_stat = corr_stat
        return corr_stat

    def getCorrStat2(self, cutoff=0.05):
        if self.corr_mean.empty:
            df = self.getCorrMean()
        else:
            df = self.corr_mean
        mean_global = round(df['corr_mean_MA(20)'].mean(), 4)
        num_global = len(df)
        df_non_zeros = df[df['corr_mean_MA(20)']>cutoff]['corr_mean_MA(20)']
        mean_non_zeros = round(df_non_zeros.mean(), 4)
        mean_non_zeros = mean_non_zeros if not math.isnan(mean_non_zeros) else 0
        num_non_zeros = len(df_non_zeros)
        df_zeros = df[df['corr_mean_MA(20)']<=cutoff]['corr_mean_MA(20)']
        mean_zeros = round(df_zeros.mean(), 4)
        num_zeros = len(df_zeros)
        ratio_non_zeros = num_non_zeros / num_global
        corr_stat2 = {'mean_global': mean_global, 'num_global': num_global, 'mean_non_zeros': mean_non_zeros, 'num_non_zeros': num_non_zeros, 'mean_zeros': mean_zeros, 'num_zeros': num_zeros, 'ratio_non_zeros': ratio_non_zeros}
        self.corr_stat2 = corr_stat2
        return corr_stat2

    def getTimelagScore(self, option=None):
        corr_stat2 = self.getCorrStat2()
        quality = int(round(corr_stat2['mean_global'], 2)*100)
        strength = int(round(corr_stat2['mean_non_zeros'], 2)*100)
        prevalence = int(round(corr_stat2['ratio_non_zeros'], 2)*100)
        total_score = round(np.mean([quality, strength, prevalence]), 2)

        if option == 'deploy':
            # quality_correction = 100 if quality*2 >= 100 else quality*2
            # strength_correction = 100 if strength*2 >= 100 else strength*2  
            volatility = int(round(self.getCorrStat()['corr_mean'].loc['std'], 2)*100)
            # total_score_correction = round(np.mean([quality_correction, strength_correction, prevalence]), 2)
            score = {'target': self.target.ticker, 'source':self.source.ticker, 'lag':self.lag, 'quality': quality, 'strength': strength, 'volatility': volatility, 'prevalence': prevalence, 'total_score':total_score}
        else: 
            score = {'target': self.target.ticker, 'source':self.source.ticker, 'lag':self.lag, 'quality': quality, 'strength': strength, 'prevalence': prevalence, 'total_score':total_score}

        return score

    def getNonZeorDates(self):
        if self.corr_stat2 == None:
            self.getCorrStat2()
            df = self.corr_mean
        else:
            df = self.corr_mean
        mean_global = self.corr_stat2['mean_global'] * 0.8
        cutoff_level = mean_global if mean_global > 0 else 0.1
        df_non_zeros = df[df['corr_mean_MA(20)']>=cutoff_level][['corr_mean_MA(20)']]
        return df_non_zeros

    def plot_nonZeros(self):
        df = self.getNonZeorDates()
        calplot.calplot(df)

    def plot_nonZeros_stramlit(self, lag):
        df = self.getNonZeorDates()
        df = df.reset_index()
        df = df.rename(columns={'index': 'date'})
        if not df.empty:
            fig = calplot(
                    df,
                    x="date",
                    y="corr_mean_MA(20)",
                    years_title =True,
                    colorscale='reds',
            )
            fig.update_layout(
                # title=f'Timelag correlation calendar: lagged {lag} source({self.source.ticker}) -> target({self.target.ticker})',
                # height=auto
                )
            st.plotly_chart(fig)
        else:
            st.write('... there is no significance in timelag correlation.')

    def plot_periodicCorr(self):
        if self.corr_mean.empty:
            df = self.getCorrMean()
        else:
            df = self.corr_mean
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(df.index, df['corr_mean'], label='Original', linewidth=1)
        ax.plot(df.index, df['corr_mean_MA(20)'], label='Moving Average (window=20)', linewidth=2)

        ax.set_xlabel('Date')
        ax.set_ylabel('corr_mean')
        ax.set_title(f'Periodic timelag correlation: lagged source({self.source.ticker})->target({self.target.ticker})')
        ax.legend()

        # x축 틱 설정
        num_ticks = 20  
        total_rows = len(df)
        stride = max(total_rows // (num_ticks - 1), 1)
        xticks = df.index[::stride].tolist()[:-1]
        xticks.append(df.index[-1])  # 마지막 날짜를 추가
        ax.set_xticks(xticks)

        plt.xticks(rotation=45)

        # 오른쪽 값에 해당하는 위치에 값을 표시하고 가로 점선 그리기
        right_value = df['corr_mean_MA(20)'].iloc[-1]
        ax.axhline(right_value, color='black', linestyle='--')
        ax.annotate(f'{right_value:.2f}', xy=(df.index[-1], right_value),
                    xytext=(20, 5), textcoords='offset points', color='black', ha='right')
        # q2_value = round(self.getCorrStat().loc['50%']['corr_mean'], 2)
        # ax.axhline(y=q2_value, color='gray', linestyle='--')
        # ax.annotate(f'Q2={q2_value}', xy=(df.index[0], q2_value), xytext=(20, 0), textcoords='offset points', color='gray', ha='left', va='center')

        plt.grid(axis='x')
        plt.show()        

    def plot_periodicCorr_streamlit(self, lag):
        if self.corr_mean.empty:
            df = self.getCorrMean()
        else:
            df = self.corr_mean
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['corr_mean'], name='Original', line=dict(width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df['corr_mean_MA(20)'], name='Moving Average (window=20)', line=dict(color='red', width=2)))

        fig.update_layout(
            xaxis=dict(title='Date'),
            yaxis=dict(title='corr_mean'),
            # title=f"Periodic timelag correlation: lagged {lag} source({self.source.ticker})->target({self.target.ticker})",
            legend=dict(x=0.02, y=0.98),
            width=800,
            # height=400
        )
        # x축 틱 설정
        num_ticks = 20
        total_rows = len(df)
        stride = max(total_rows // (num_ticks - 1), 1)
        xticks = df.index[::stride].tolist()[:-1]
        xticks.append(df.index[-1])  # 마지막 날짜를 추가
        fig.update_xaxes(tickmode='array', tickvals=xticks, tickangle=45)

        # 오른쪽 값에 해당하는 위치에 값을 표시하고 가로 점선 그리기
        # right_value = df['corr_mean_MA(20)'].iloc[-1]
        mean_non_zeros = self.getCorrStat2()['mean_non_zeros']
        right_value = mean_non_zeros 

        fig.add_shape(
            type='line',
            x0=df.index[0],
            y0=right_value,
            x1=df.index[-1],
            y1=right_value,
            line=dict(color='gray', dash='dash')
        )
        fig.add_annotation(
            x=df.index[-1],
            y=right_value,
            text=f"{right_value:.2f}",
            xanchor='left',
            yanchor='middle',
            showarrow=False,
            font=dict(color='gray')
        )

        st.plotly_chart(fig)

    def rolling(self, roll_range, bidirect=True, winsor=True, limit=0.05, solver='return'):
        if solver == 'return':
            self.solve = self.corr
        elif solver == 'price':
            self.solve = self.corr_price
        elif solver == 'volume':
            self.solve = self.corr_vol
        elif solver == 'ma5':
            self.solve = self.corr_ma5
        elif solver == 'dtw':
            self.solve = self.dtw

        start_time = time.time()
        print(f"start: rolling ...")

        j = -roll_range
        bucket = []
        self.lag = self.solution()['lag']
        for i in range(roll_range+1):
            bucket.append({'shift': i+j, 'corr': self.set_shift(i+j).solve(winsor=winsor, limit=limit)['corr'], 'period': self.period})
        
        if bidirect == True:
            bucket_future = []
            for k in range(roll_range):
                bucket_future.append({'shift': k+1, 'corr': self.set_shift(k+1).solve(winsor=winsor, limit=limit)['corr'], 'period': self.period})
            bucket = bucket + bucket_future

        self.df_roll = pd.DataFrame(bucket)
        self.set_shift(0)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"end: {round(execution_time, 2)} seconds elapsed")

        return self.df_roll
    
    def plot_rolling(self, roll_range, bidirect=True, winsor=True, limit=0.05, solver='return'):

        text_period = self.period
        text_lag = self.solution(solver=solver)['lag']

        df_roll = self.rolling(roll_range, bidirect, winsor=winsor, limit=limit, solver=solver)
        shift_values = df_roll['shift'].tolist()
        corr_values = df_roll['corr'].tolist()
        self.set_lag(0)

        plt.figure(figsize=(12, 6))
        plt.plot(shift_values, corr_values, marker='o')
        
        plt.axhline(y=corr_values[shift_values.index(0)], color='r', linestyle='--')
        plt.xlabel('shift (day)')
        plt.ylabel('corr')
        plt.title(f'Rolling: lag {text_lag}, range {roll_range}, period {text_period}, source({self.source.ticker}) -> target({self.target.ticker})')
        
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_position(('data', 0))
        plt.grid(axis='x')
        plt.show()

    def validate_timelag(self, roll_range, cutoff=0.6):
        self.roll(roll_range)
        df = self.df_roll
        max_corr = df['corr'].max()
        df['corr_normalized'] = df['corr'] / max_corr
        df['corr_diff'] = df['corr_normalized'].diff().abs().fillna(0)
        isValid = all(df['corr_diff'] < cutoff)
        self.df_roll = df
        return isValid 

    def plot_compare(self, best=False, winsor=True, limit=0.05, solver='return'):
        if best == True:
            solution = self.solution(winsor=winsor, limit=limit, solver=solver)            
        else:
            if solver == 'return':
                self.solve = self.corr
            elif solver == 'price':
                self.solve = self.corr_price
            elif solver == 'volume':
                self.solve = self.corr_vol
            elif solver == 'ma5':
                self.solve = self.corr_ma5
            elif solver == 'dtw':
                self.solve = self.dtw
            solution = self.solve(winsor=winsor, limit=limit)

        sol_corr = round(solution['corr'], 3)
        sol_lag = solution['lag']
        sol_date_target = solution['date_target']
        sol_date_source = solution['date_source']
        self.set_lag(sol_lag)

        title_header = f'Corr = {sol_corr}' if best==True else f'Corr = {sol_corr}, shfit = {self.shift}'

        price_compare = pd.DataFrame()
        price_compare['price_target'] = (self.data_target[self.data_target['date'] <= sol_date_target]['PX_LAST_t'].iloc[:self.period]).tolist()
        price_compare['price_source'] = (self.data_source[self.data_source['date'] <= sol_date_source]['PX_LAST_s'].iloc[:self.period]).tolist()
        price_compare['normed_price_target'] = (price_compare['price_target']/price_compare['price_target'].iloc[0]-1)*100
        price_compare['normed_price_source'] = (price_compare['price_source']/price_compare['price_source'].iloc[0]-1)*100
        self.price_compare = price_compare
        
        # 마지막 데이터 포인트의 값을 가져옴
        last_target_price = price_compare['normed_price_target'].iloc[-1]
        last_source_price = price_compare['normed_price_source'].iloc[-1]

        # 플롯에 텍스트로 마지막 데이터 포인트의 값을 표시
        plt.text(len(price_compare.index) - 1, last_target_price, f'{last_target_price:.3f}', ha='right', va='center')
        plt.text(len(price_compare.index) - 1, last_source_price, f'{last_source_price:.3f}', ha='right', va='center')

        plt.plot(price_compare.index, price_compare['normed_price_target'], label='target')
        plt.plot(price_compare.index, price_compare['normed_price_source'], label=f'lagged {self.lag} source')
        plt.xlabel(f'Day (period: {self.period})')
        plt.ylabel('Cumulative return (%)')
              
        plt.title(f'{title_header}: {self.target.ticker} vs. lagged {self.lag} {self.source.ticker}')
        plt.xticks(np.arange(0, len(price_compare.index), 5))
        plt.legend()
        plt.grid(axis='x')
        plt.show()
        

    def plot_compare_streamlit(self, best=False, winsor=True, limit=0.05, solver='return'):
        if best == True:
            solution = self.solution(winsor=winsor, limit=limit, solver=solver)            
        else:
            if solver == 'return':
                self.solve = self.corr
            elif solver == 'price':
                self.solve = self.corr_price
            elif solver == 'volume':
                self.solve = self.corr_vol
            elif solver == 'ma5':
                self.solve = self.corr_ma5
            elif solver == 'dtw':
                self.solve = self.dtw
            solution = self.solve(winsor=winsor, limit=limit)

        sol_corr = round(solution['corr'], 3)
        sol_lag = solution['lag']
        sol_date_target = solution['date_target']
        sol_date_source = solution['date_source']
        self.set_lag(sol_lag)

        title_header = f'Corr = {sol_corr}'

        price_compare = pd.DataFrame()
        price_compare['price_target'] = (self.data_target[self.data_target['date'] <= sol_date_target]['PX_LAST_t'].iloc[:self.period]).tolist()
        price_compare['price_source'] = (self.data_source[self.data_source['date'] <= sol_date_source]['PX_LAST_s'].iloc[:self.period]).tolist()
        price_compare['normed_price_target'] = (price_compare['price_target']/price_compare['price_target'].iloc[0]-1)*100
        price_compare['normed_price_source'] = (price_compare['price_source']/price_compare['price_source'].iloc[0]-1)*100
        self.price_compare = price_compare

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price_compare.index, y=price_compare['normed_price_target'], name='target', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=price_compare.index, y=price_compare['normed_price_source'], name=f'lagged {self.lag} source', line=dict(color='green')))
        fig.update_layout(
            title=f'{title_header}: lagged {self.lag} source({self.source.ticker}) >> target({self.target.ticker})',
            xaxis=dict(title=f'Day (period: {self.period})'),
            yaxis=dict(title='Cumulative return (%)'),
            xaxis_tickvals=list(price_compare.index[::5]),
            legend=dict(x=0, y=1)
        )

        # 마지막 데이터 포인트의 값을 가져옴
        last_target_price = price_compare['normed_price_target'].iloc[-1]
        last_source_price = price_compare['normed_price_source'].iloc[-1]

        # 플롯에 텍스트로 마지막 데이터 포인트의 값을 표시
        fig.add_annotation(
            x=len(price_compare.index) - 1,
            y=last_target_price,
            text=f'{last_target_price:.3f}',
            showarrow=False,
            font=dict(color='black', size=12),
            align='right',
            xshift=-5,
            yshift=5
        )
        fig.add_annotation(
            x=len(price_compare.index) - 1,
            y=last_source_price,
            text=f'{last_source_price:.3f}',
            showarrow=False,
            font=dict(color='black', size=12),
            align='right',
            xshift=-5,
            yshift=5
        )
        st.plotly_chart(fig)


    def plot_returns(self, best=True, winsor=True, limit=0.05, solver='return'):
        if best == True:
            solution = self.solution(winsor=winsor, limit=limit, solver=solver)            
        else:
            if solver == 'return':
                self.solve = self.corr
            elif solver == 'price':
                self.solve = self.corr_price
            elif solver == 'volume':
                self.solve = self.corr_vol
            elif solver == 'ma5':
                self.solve = self.corr_ma5
            elif solver == 'dtw':
                self.solve = self.dtw
            solution = self.solve(winsor=winsor, limit=limit)

        sol_corr = round(solution['corr'], 3)
        sol_lag = solution['lag']
        sol_date_target = solution['date_target']
        sol_date_source = solution['date_source']
        self.set_lag(sol_lag)

        r = pd.DataFrame()
        r['CHG_PCT_1D_t'] = (self.data_target[self.data_target['date'] <= sol_date_target]['CHG_PCT_1D_t'].iloc[:self.period]).tolist()
        r['CHG_PCT_1D_s'] = (self.data_source[self.data_source['date'] <= sol_date_source]['CHG_PCT_1D_s'].iloc[:self.period]).tolist()
        r['delta'] = r.apply(lambda x: 0 if x['CHG_PCT_1D_t'] * x['CHG_PCT_1D_s'] > 0 else None, axis=1)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        ax1.bar(r.index, r['CHG_PCT_1D_t'], color='blue', alpha=0.7, label='target')
        ax1.bar(r.index, r['CHG_PCT_1D_s'], color='green', alpha=0.7, label='source')

        ax1.set_xlabel(f'Day (period: {self.period})')
        ax1.set_ylabel('Return (%)')
        ax1.set_title('Return Comparison')
        ax1.legend()
        ax1.grid(True, axis='x')

        same_sign_indices = (r['CHG_PCT_1D_t'] * r['CHG_PCT_1D_s'] > 0)
        ax2.scatter(r.loc[same_sign_indices].index, [0] * sum(same_sign_indices),
                    color='red', marker='o', s=50)

        ax2.set_xlabel(f'Day (period: {self.period})')
        ax2.set_title(f'Same Sign Indicator (ratio: {sum(r["delta"] == 0) / len(r):.2%})')
        ax2.grid(True, axis='x')

        plt.setp(ax1.get_xticklabels(), rotation=45)
        plt.setp(ax2.get_xticklabels(), rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_returns_streamlit(self, best=True, winsor=True, limit=0.05, solver='return'):
        if best == True:
            solution = self.solution(winsor=winsor, limit=limit, solver=solver)            
        else:
            if solver == 'return':
                self.solve = self.corr
            elif solver == 'price':
                self.solve = self.corr_price
            elif solver == 'volume':
                self.solve = self.corr_vol
            elif solver == 'ma5':
                self.solve = self.corr_ma5
            elif solver == 'dtw':
                self.solve = self.dtw
            solution = self.solve(winsor=winsor, limit=limit)

        sol_corr = round(solution['corr'], 3)
        sol_lag = solution['lag']
        sol_date_target = solution['date_target']
        sol_date_source = solution['date_source']
        self.set_lag(sol_lag)

        r = pd.DataFrame()
        r['CHG_PCT_1D_t'] = (self.data_target[self.data_target['date'] <= sol_date_target]['CHG_PCT_1D_t'].iloc[:self.period]).tolist()
        r['CHG_PCT_1D_s'] = (self.data_source[self.data_source['date'] <= sol_date_source]['CHG_PCT_1D_s'].iloc[:self.period]).tolist()
        r['delta'] = r.apply(lambda x: 0 if x['CHG_PCT_1D_t'] * x['CHG_PCT_1D_s'] > 0 else None, axis=1)

        fig = go.Figure()

        # Add bar trace for target returns
        fig.add_trace(go.Bar(
            x=r.index,
            y=r['CHG_PCT_1D_t'],
            name='target',
            marker=dict(color='blue', opacity=0.7)
        ))

        # Add bar trace for source returns
        fig.add_trace(go.Bar(
            x=r.index,
            y=r['CHG_PCT_1D_s'],
            name='source',
            marker=dict(color='green', opacity=0.7)
        ))

        # Configure layout for the bar chart
        fig.update_layout(
            xaxis=dict(title=f'Day (period: {self.period})'),
            yaxis=dict(title='Return (%)'),
            title='Return Comparison',
            barmode='overlay',
            legend=dict(orientation='h'),
            xaxis2=dict(showgrid=True),
            yaxis2=dict(showgrid=False),
            plot_bgcolor='white'
        )

        # Add scatter trace for same sign indicator
        same_sign_indices = (r['CHG_PCT_1D_t'] * r['CHG_PCT_1D_s'] > 0)
        fig.add_trace(go.Scatter(
            x=r.loc[same_sign_indices].index,
            y=[0] * sum(same_sign_indices),
            mode='markers',
            marker=dict(color='red', size=4),
            name='Indicator'
        ))

        # Configure layout for the scatter plot
        fig.update_layout(
            xaxis2=dict(domain=[0, 1], showticklabels=False),
            yaxis2=dict(domain=[0.25, 0.3], showticklabels=False),
            annotations=[
                dict(
                    x=0.5,
                    y=0.1,
                    text=f"Same Sign Indicator (ratio: {sum(r['delta'] == 0) / len(r):.2%})",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    font=dict(size=12)
                )
            ]
        )
        fig = go.Figure(fig)
        st.plotly_chart(fig)

    def plot_price(self, option="price"):
        df_ref = self.df
        if option == "price":
            df = df_ref[['date', 'PX_LAST_t', 'PX_LAST_s']]
            title = 'Price'
        if option == "ma":
            df = df_ref[['date', 'MOV_AVG_5D_t', 'PX_MOV_AVG_5D_s']]
            title = 'MA(5)'
        df.set_index('date', inplace=True)

        dict_y_unit = {'KOSPI200': '(₩)', 'KOSDAQ': '(₩)', 'RUSSELL3000': '($)', 'PORTFOLIO': ''}

        fig, ax1 = plt.subplots(figsize=(10,6))      
        ax1.set_ylabel(f'price_target {dict_y_unit[self.target.marketName]}')
        ax1.plot(df.index, df.iloc[:, 0], color='C0')
        ax2 = ax1.twinx()
        ax2.set_ylabel(f'price_source {dict_y_unit[self.source.marketName]}')
        ax2.plot(df.index, df.iloc[:, 1], color='C1')
        ax1.legend(['target stock'], loc='upper left')
        ax2.legend(['source stock'], loc='upper right')

        # num_ticks = 10
        # x_ticks = np.linspace(0, len(df)-1, num_ticks, dtype=np.int64)
        # ax1.set_xticks(x_ticks)
        # ax1.set_xticklabels(df.index[x_ticks], rotation=45, ha='right')

        plt.title(f'{title}- target: {self.target.ticker}, source: {self.source.ticker}')
        plt.show()


    def plot_price_streamlit(self, option="price"):
        df_ref = self.df
        if option == "price":
            df = df_ref[['date', 'PX_LAST_t', 'PX_LAST_s']]
            title = 'Price'
        if option == "ma":
            df = df_ref[['date', 'MOV_AVG_5D_t', 'PX_MOV_AVG_5D_s']]
            title = 'MA(5)'
        df.set_index('date', inplace=True)

        dict_y_unit = {'KOSPI200': '(₩)', 'KOSDAQ': '(₩)', 'RUSSELL3000': '($)', 'PORTFOLIO': ''}

        # fig = go.Figure()
        # fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, 0], name='target stock', line=dict(color='blue')))
        # fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, 1], name='source stock', line=dict(color='red')))

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        fig.add_trace(
            go.Scatter(x=df.index, y=df.iloc[:, 0], line=dict(color='blue'), name="target"),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=df.index, y=df.iloc[:, 1], line=dict(color='green'), name="source"),
            secondary_y=True,
        )


        fig.update_layout(
            title=f'{title}: target({self.target.ticker}) vs. source({self.source.ticker})',
            xaxis=dict(title='Date'),
            yaxis=dict(title=f'price_target {dict_y_unit[self.target.marketName]}'),
            yaxis2=dict(title=f'price_source {dict_y_unit[self.source.marketName]}', side='right', overlaying='y'),
            legend=dict(x=0.02, y=0.98)
        )

        st.plotly_chart(fig)


    def plot_stat(self, solver='return'):
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
        df = self.result(solver)

        # 첫 번째 플롯 그리기
        axes[0].plot(df['lag'], df['corr'], color='red')
        axes[0].set_xlabel('time lag')
        axes[0].set_ylabel('correlation')
        axes[0].set_title(f'Corr by lag: source({self.source.ticker}) -> target({self.target.ticker})')

        max_corr = df['corr'].max()
        min_corr = df['corr'].min()
        axes[0].annotate(f'Max: {max_corr:.2f}', xy=(df['lag'].iloc[df['corr'].idxmax()], max_corr),
                         xytext=(5, 0), textcoords='offset points', color='red')
        axes[0].annotate(f'Min: {min_corr:.2f}', xy=(df['lag'].iloc[df['corr'].idxmin()], min_corr),
                         xytext=(5, -10), textcoords='offset points', color='blue')

        mean_corr = df['corr'].mean()
        axes[0].axhline(mean_corr, color='red', linestyle='dashed', linewidth=0.5)
        axes[0].annotate(f'Mean: {mean_corr:.2f}', xy=(0, mean_corr),
                     xytext=(5, 0), textcoords='offset points', color='gray')
        
        # 두 번째 플롯 그리기
        axes[1].plot(df['lag'], df['p_value'], color='gray')
        axes[1].set_xlabel('time lag')
        axes[1].set_ylabel('p-value')
        axes[1].set_title(f'p-value by lag: source({self.source.ticker}) -> target({self.target.ticker})')

        mean_corr = df['p_value'].mean()
        axes[1].axhline(mean_corr, color='gray', linestyle='dashed', linewidth=0.5)
        axes[1].annotate(f'Mean: {mean_corr:.2f}', xy=(0, mean_corr),
                     xytext=(5, 0), textcoords='offset points', color='gray')
        
        max_p = df['p_value'].max()
        min_p = df['p_value'].min()
        axes[1].annotate(f'Min: {min_p:.2f}', xy=(df['lag'].iloc[df['p_value'].idxmin()], min_p),
                         xytext=(5, -10), textcoords='offset points', color='blue')

        plt.tight_layout()
        plt.show()


    def plot_stat_streamlit(self, solver='return'):
        fig = sp.make_subplots(rows=2, cols=1, subplot_titles=(f'Corr by lag: source({self.source.ticker}) -> target({self.target.ticker})',
                                                            f'p-value by lag: source({self.source.ticker}) -> target({self.target.ticker})'))

        df = self.result(solver)

        # 첫 번째 플롯 그리기
        fig.add_trace(go.Scatter(x=df['lag'], y=df['corr'], mode='lines', line=dict(color='red')),
                    row=1, col=1)
        fig.update_xaxes(title_text='time lag', row=1, col=1)
        fig.update_yaxes(title_text='correlation', row=1, col=1)

        max_corr = df['corr'].max()
        min_corr = df['corr'].min()
        fig.add_annotation(x=df['lag'].iloc[df['corr'].idxmax()], y=max_corr, text=f'Max: {max_corr:.2f}', showarrow=True,
                        arrowhead=1, arrowsize=2, arrowcolor='red', ax=5, ay=0)
        fig.add_annotation(x=df['lag'].iloc[df['corr'].idxmin()], y=min_corr, text=f'Min: {min_corr:.2f}', showarrow=True,
                        arrowhead=1, arrowsize=2, arrowcolor='blue', ax=5, ay=-10)

        mean_corr = df['corr'].mean()
        fig.add_shape(type='line', x0=df['lag'].min(), y0=mean_corr, x1=df['lag'].max(), y1=mean_corr,
                    line=dict(color='red', dash='dash'), row=1, col=1)
        fig.add_annotation(x=0, y=mean_corr, text=f'Mean: {mean_corr:.2f}', showarrow=False,
                        xanchor='left', yanchor='middle', xshift=5, yshift=0, font=dict(color='gray'))

        # 두 번째 플롯 그리기
        fig.add_trace(go.Scatter(x=df['lag'], y=df['p_value'], mode='lines', line=dict(color='gray')),
                    row=2, col=1)
        fig.update_xaxes(title_text='time lag', row=2, col=1)
        fig.update_yaxes(title_text='p-value', row=2, col=1)

        mean_p = df['p_value'].mean()
        fig.add_shape(type='line', x0=df['lag'].min(), y0=mean_p, x1=df['lag'].max(), y1=mean_p,
                    line=dict(color='gray', dash='dash'), row=2, col=1)
        fig.add_annotation(x=0, y=mean_p, text=f'Mean: {mean_p:.2f}', showarrow=False,
                        xanchor='left', yanchor='middle', xshift=5, yshift=0, font=dict(color='gray'))

        max_p = df['p_value'].max()
        min_p = df['p_value'].min()
        fig.add_annotation(x=df['lag'].iloc[df['p_value'].idxmin()], y=min_p, text=f'Min: {min_p:.2f}', showarrow=True,
                        arrowhead=1, arrowsize=2, arrowcolor='blue', ax=5, ay=-10)

        fig.update_layout(title_text='Stat Plots', height=800)

        st.plotly_chart(fig)


class Market:
    def __init__(self, marketName, sectorName=None):
        self.marketName = marketName
        self.df = self.df()
        self.tickers = self.df['value'].tolist()
        if sectorName == None:
            self.members = self.setMembers()
        else:
            self.members = self.setSectorMembers(sectorName=sectorName)

    def df(self):
        folder_path = f'./datasets-member/'
        df = pd.read_csv(folder_path+ f'dataset-{self.marketName}-members.csv', ) 
        return df

    def getGICSInfo(self):
        folder_path = f'./datasets-gics/'
        df = pd.read_csv(folder_path+ f'dataset-{self.marketName}-gics.csv') 
        df = df.set_index('ticker', inplace=False)
        self.gics_info = df
        return self.gics_info

    def setSectorMembers(self, category='GICS_SECTOR_NAME', sectorName='Industrials'):
        df = self.getGICSInfo()
        tickers_by_sector = df[df[category]==sectorName].index
        members_by_sector = [Stock(ticker, self.marketName) for ticker in tickers_by_sector]     
        return members_by_sector

    def searchMember(self, keyword=None, index=None):
        if keyword != None and index == None:
            member = self.df[self.df['value'].str.contains(keyword)==True]
        elif keyword == None and index != None:
            member = self.df[self.df.index==index]
        return member

    def setMembers(self):
        tickers = self.tickers
        # 코스닥의 경우 신규 편입 마지막 두 종목 데이터 없음: 454640 KS. 455250 KS 
        if self.marketName == 'KOSDAQ':
            tickers = tickers[:2]
        members = [Stock(ticker, self.marketName) for ticker in tickers]
        return members

    def oneMember(self, index=None):
        n = len(self.df)
        if index == None:
            index = random.randint(0, n-1)
        return self.setMembers()[index]

    def getCountInfo(self, category='GICS_SECTOR_NAME'):
        df = self.getGICSInfo()
        f"""
        category (str): {self.gics_info.columns}

        Returns:
        결과 (str): 작업의 결과를 문자열 형태로 반환합니다.
        """
        df_counts = df[category].value_counts().reset_index()
        df_counts.columns = [category, 'count']

        # 히스토그램 그리기
        plt.figure(figsize=(10, 6))
        plt.bar(df_counts[category], df_counts['count'])
        plt.xlabel(category)
        plt.ylabel('count')
        plt.title(f'Count by {category}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        return df_counts

    def makeBatch(self, batch_size):
        bucket = []
        q = len(self.df) // batch_size
        for i in range(q+1):
            index_i = i*batch_size
            index_f = (i+1)*batch_size if i != q else None
            bucket.append(self.members[index_i:index_f])
        if bucket[-1] == []:
            bucket = bucket[:-1]

        self.batches = bucket
        return self.batches


class InterMarket:
    def __init__(self, target_market, source_market):
        self.target_market = target_market
        self.source_market = source_market 

    def getPairs(self):
        if isinstance(self.target_market, Market):
            targets = self.target_market.members
        else:
            targets = self.target_market
        if isinstance(self.source_market, Market):
            sources = self.source_market.members
        else:
            sources = self.source_market

        bucket = []
        for target in targets:
            for source in sources:
                if target != source:
                    bucket.append(Pair(target, source))
        return bucket

    def solve_timelags(self, solver='return', winsor=True, limit=0.05, target_index=0, batch_size=100000, batch_num=1, save=True):
        start_time = time.time()
        print(f"start: time lag solutions ...")

        target = self.target_market.members[target_index]
        source_batches = self.source_market.makeBatch(batch_size)[:batch_num] 
        num_of_batches = len(source_batches)

        bucket = []
        for source_batch in source_batches:
            n = len(source_batch)
            i = 1
            for source in source_batch:
                print(f"-- ({i}/{n}) pair: (target: {target.ticker}, source: {source.ticker}) ...")
                pair = Pair(target, source)
                solution = pair.solution(winsor=winsor, limit=limit, solver=solver)
                bucket.append(solution)
                i += 1

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"end: {round(execution_time, 2)} seconds elapsed")

        self.result = pd.DataFrame(bucket)

        if save == True:
            text_ticker = target.ticker.replace('/', '_')
            text_batch = '0' + str(batch_num) if batch_num < 10 else str(batch_num)
            subname = f'result-{text_ticker}-batch-' + text_batch
            self.saveResult(self.result, subname)

        return self.result

    def solve_timelags_parallel(self, solver='return', winsor=True, limit=0.05, target_index=0, batch_size=100000, batch_num=1, save=True):
        start_time = time.time()
        print(f"start: time lag solutions ...")

        target = self.target_market.members[target_index]
        sources = self.source_market.members 
        n = len(sources)

        def f(i, source):
            print(f"-- ({i}/{n}) pair: (target: {target.ticker}, source: {source.ticker}) ...")
            pair = Pair(target, source)
            solution = pair.solution(winsor=winsor, limit=limit, solver=solver)        
            return solution

        with Parallel(n_jobs=mp.cpu_count()) as parallel:
            solutions = parallel(delayed(f)(i, source) for (i, source) in zip(range(1,n+1), sources))

        self.result = pd.DataFrame(solutions)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"end: {round(execution_time, 2)} seconds elapsed")

        if save == True:
            text_ticker = target.ticker.replace('/', '_')
            text_batch = '0' + str(batch_num) if batch_num < 10 else str(batch_num)
            subname = f'result-{text_ticker}-batch-' + text_batch
            self.saveResult(self.result, subname)

        return self.result

    def analyze_corrmatrices_parallel(self, target_ticker=None, target_index=0, lag=1, min_period=5, max_period=90, max_shift=250*4, stride=10, save_time=True, task_memo=None):
        task_memo = f'-{task_memo}' if task_memo != None else ''
        time_memo = f'-{record_time()}' if save_time == True else ''
        folder_name = f"results-corrmatrix{time_memo}{task_memo}"
        folder_path = f'./{folder_name}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        target = self.target_market.members[target_index]
        if target_ticker != None and 'KS' in target_ticker:
            target = K200stock(target_ticker)
        elif target_ticker != None and 'U' in target_ticker:
            target = R3000stock(target_ticker)
        sources = self.source_market.members 
        n = len(sources)
        print(f"start: {target.ticker} x {n} correlation matrices ...")

        def f(i, source, lag=lag, max_period=max_period, max_shift=max_shift, stride=stride, folder_name=folder_name):
            print(f"-- ({i}/{n}) pair: (target: {target.ticker}, source: {source.ticker}) ...")
            pair = Pair(target, source)
            pair.corrmatrix(lag=lag, min_period=min_period, max_period=max_period, max_shift=max_shift, stride=stride, plot=False, folder_name=folder_name)      
        
        with Parallel(n_jobs=mp.cpu_count()) as parallel:
            parallel(delayed(f)(i, source) for (i, source) in tqdm(zip(range(1,n+1), sources)))

    def solve_timelags_sector(self, category='GICS_SECTOR_NAME', sectorName='Industrials', solver='return', winsor=True, limit=0.05, target_index=0, batch_size=100000, batch_num=1, save=True):
        start_time = time.time()
        print(f"start: time lag solutions ...")
        
        target_stocks = self.target_market.setSectorMembers(category=category, sectorName=sectorName)
        source_stocks = self.source_market.setSectorMembers(category=category, sectorName=sectorName)

        bucket = []
        n = len(target_stocks)*len(source_stocks)
        for target in target_stocks:
            i = 1
            for source in source_stocks:
                print(f"-- ({i}/{n}) pair: (target: {target.ticker}, source: {source.ticker}) ...")
                pair = Pair(target, source)
                solution = pair.corrmatrix(winsor=winsor, limit=limit, solver=solver)
                bucket.append(solution)
                i += 1
        
        result = pd.DataFrame(bucket)
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"end: {round(execution_time, 2)} seconds elapsed")

        if save == True:
            text_ticker = target.ticker.replace('/', '_')
            subname = f'result-sector-{sector_name}'
            self.saveResult(result, subname)

        return result

    def analyze_intermarket(self, lag, min_period=20, max_period=1000, max_shift=1000, stride=5, save=True, folder_subname=None):
        start_time = time.time()
        print(f"start: analyze intermarket ...")

        pairs = self.getPairs()
        
        def f(pair, lag=lag, min_period=min_period, max_period=max_period, max_shift=max_shift, stride=stride):
            print(f"-- pair: (target: {pair.target.ticker}, source: {pair.source.ticker}) ...")
            pair.openCorrmatrixFile(lag=lag, min_period=min_period, max_period=max_period, max_shift=max_shift, stride=shift, folder_subname=folder_subname)
            return pair.getTimelagScore(option=None)

        with Parallel(n_jobs=mp.cpu_count()) as parallel:
            results = parallel(delayed(f)(pair) for pair in tqdm(pairs))

        df = pd.DataFrame(results)

        if save == True:
            folder_subname = self.record_time() if folder_subname==None else folder_subname
            folder_name = f"results-corrmatrix-{folder_subname}"
            path = f'./{folder_name}'

            if not os.path.exists(path):
                os.makedirs(path)

            file_name = f'result-timelag-analysis-lag{lag}'
            df.to_csv(f'{path}/{file_name}.csv', index=False)
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"end: {round(execution_time, 2)} seconds elapsed")

        return df

    def saveResult(self, df, subname):
        df.to_csv(f'{subname}.csv', index=False)
        print(f'saved batch: {subname}')


class Portfolio:
    def __init__(self, stocks=None, tickers=None, date=None, weights=None):
        if tickers != None:
            stocks = []
            for ticker in tickers:
                if ' KS' in ticker:
                    stock = K200stock(ticker)
                elif ' U' in ticker:
                    stock = R3000stock(ticker) 
                stocks.append(stock)
        self.marketName = 'PORTFOLIO'
        self.members = stocks
        list_ticker = [stock.ticker for stock in self.members]
        self.ticker = f'PORT({len(list_ticker)}: {list_ticker[0]} & ...)'
        self.date = date
        # weights[-1] = 1 - np.sum(weights[:-1])
        self.weights = weights
        self.df = self.df() 

    def df(self):
        stocks = self.members
        df_rs = []
        for stock in stocks:
            df_r = stock.df[['date', 'PX_LAST', 'VOLUME', 'MOV_AVG_5D', 'CHG_PCT_1D']]
            df_r = df_r.rename(columns={
                'PX_LAST': f'PX_LAST_{stock.ticker}',
                'VOLUME': f'VOLUME_{stock.ticker}',
                'MOV_AVG_5D': f'MOV_AVG_5D_{stock.ticker}',
                'CHG_PCT_1D': f'CHG_PCT_1D_{stock.ticker}',
                })
            df_rs.append(df_r)
        # df_rs = [stock.df[['date', 'CHG_PCT_1D']] for stock in stocks]
        df = df_rs[0]
        for df_r in df_rs[1:]:
            df = pd.merge(df, df_r, on='date', how='outer')
            df = df.sort_values('date')
        cols = df.columns
        col_CHG = [col for col in cols if 'CHG' in col] 
        col_not_CHG = [col for col in df.columns if 'CHG' not in col]
        df[col_CHG] = df[col_CHG].fillna(0)
        df[col_not_CHG] = df[col_not_CHG].fillna(method='ffill')
        # print(df)
        df_ref = df[df['date']<=self.date] if self.date != None else df
        date_ref = df_ref.iloc[-1]['date']        
        self.date_ref = date_ref

        # df_caps = [stock.df[['date', 'CUR_MKT_CAP']] for stock in stocks]
        # df_c = df_caps[0]
        # for df_cap in df_caps[1:]:
        #     df_c = pd.merge(df_c, df_cap, on='date', how='outer')
        #     df_c = df_c.sort_values('date')
        #     df_c = df_c.fillna(method='ffill')        

        # 환율 계산 반영 추가할 것
        # ex_ref = df_ex[df_ex['date']==self.date_ref]['exchane_rate'] 

        if self.weights != None and sum(self.weights) != 1:
            raise ValueError("(input error) sum of weights must be 1")
        elif self.weights == None:
            caps = df_c[df_c['date']==date_ref].to_dict(orient='split')['data'][0][1:]
            self.capitalizations = caps
            caps_tot = sum(caps)
            ws = [cap/caps_tot for cap in caps]
            self.weights = ws

        cols = df.columns
        col_PX = [col for col in cols if 'PX' in col]
        df['PX_LAST'] = sum([df[col_PX].iloc[:, i] * self.weights[i] for i in range(len(self.weights))])
        col_VOL = [col for col in cols if 'VOL' in col] 
        df['VOLUME'] = sum([df[col_VOL].iloc[:, i] * self.weights[i] for i in range(len(self.weights))])
        col_MOV = [col for col in cols if 'MOV' in col] 
        df['MOV_AVG_5D'] = sum([df[col_MOV].iloc[:, i] * self.weights[i] for i in range(len(self.weights))])
        col_CHG = [col for col in cols if 'CHG' in col] 
        df['CHG_PCT_1D'] = sum([df[col_CHG].iloc[:, i] * self.weights[i] for i in range(len(self.weights))])
        rs = df['CHG_PCT_1D']/100 + 1
        rs_corrected = np.array(rs[:-1])
        rs_corrected = np.insert(rs_corrected, 0, 1)
        cumulatives = np.cumprod(rs_corrected)
        df['normed_price'] = cumulatives
        df['return_cumulative'] = round((df['normed_price'] - 1)*100, 5)

        df = df.reindex(columns=df.columns[:1].append(df.columns[-6:].append(df.columns[1:-6])))
        self.df = df

        return df
        

def solve_timelags(target, source_market):
    start_time = time.time()
    print(f"start: time lag solutions ...")
    
    ticker_target = target.ticker

    n = len(source_market.df)
    i = 1
    bucket = []
    for source in source_market.members:
        print(f"-- ({i}/{n}) pair: (target: {ticker_target}, source: {source.ticker}) ...")
        pair = Pair(target, source)
        solution = pair.solution()
        bucket.append(solution)
        i += 1

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"end: {execution_time} seconds elapsed")

    result = pd.DataFrame(bucket)
    return result


def getRankedPair(df, rank=1):
    i = rank - 1 
    row = df.iloc[i]
    t = K200stock(row['target'])
    s = R3000stock(row['source'])
    pair = Pair(t, s)
    return pair


def saveResult(df_result, subname):
    df_result.to_csv(f'{subname}-batch-result.csv', index=False)
    print(f'saved batch: {subname}')
    print('')    


def openRankedResult(df):
    df = df.sort_values(by='corr', key=abs, ascending=False)
    df = df.reset_index(drop=True)
    return df


def mergeResultFiles(folder_path, file_name):
    file_names = os.listdir(folder_path)
    dataframes = []
    for name in file_names:
        path = os.path.join(folder_path, name)
        _, file_ext = os.path.splitext(name)
        if file_ext.lower() == '.csv':
            df = pd.read_csv(path)            
            dataframes.append(df)
    
    merged_df = pd.concat(dataframes, ignore_index=True)    
    merged_df.to_csv(file_name, index=False)    
    return merged_df


def getImportTime():
    current_datetime = datetime.now()
    print('Module [interMarket Analyzer] imported at:', current_datetime)


def getBulkAnswers(stocks, start_index=0, end_index=None):
    last_index = len(stocks)-1
    if end_index != None and last_index < end_index:
        end_index = None 
    print('start LLM:gpt')
    start_time = time.time()
    bucket = []
    n = len(stocks[start_index:end_index])
    i = 1
    for stock in stocks[start_index:end_index]:
        print(f'({i}/{n}) {stock.ticker} ...')
        try:
            info = stock.generateGPTanswer()
        except Exception as err:
            print(f'{stock.ticker}: {err}')
            info = {'ticker': stock.ticker }
        bucket.append(info)
        i += 1

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('end.') 
    print('total elapsed time: %.2f seconds' %elapsed_time)
    return pd.DataFrame(bucket)


def getNullAnswers(marketName, folder_path):
    ans = pd.read_csv(folder_path + f'dataset-{marketName}-gpt.csv')
    tickers_nan = ans[ans['NAME'].isna() == True].ticker
    stocks = [Stock(x, marketName) for x in tickers_nan]
    df_result_nans = ia.getBulkAnswers(stocks)
    df_result_nans.to_csv(f'dataset-{marketName}-gpt-nulls.csv', index=False)


def record_time(option=None):
    now = datetime.now()
    form = "%Y%m%d%H%M" if option=='min' else "%Y%m%d%H"
    formatted_date = now.strftime(form)
    return formatted_date


def download_marketMembers(con, marketName):
    info = {'KOSPI200': 'KOSPI2', 'RUSSELL3000': 'RAY', 'KOSDAQ':'KOSDAQ'}
    df = con.bulkref(f'{info[marketName]} Index', 'INDX_MEMBERS')
    if marketName == 'RUSSELL3000':
        df2 = con.bulkref(f'{info[marketName]} Index', 'INDX_MEMBERS2')
        df = pd.concat([df, df2], axis=0)
        df.reset_index(inplace=True)

    folder_name = f"downloads-members"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    df.to_csv(f'./{folder_name}/dataset-{marketName}-members-{record_time()}.csv', index=False)

    return df

def openCSV(file_path, index_col='period'):
    return pd.read_csv(file_path, index_col=index_col)

def saveCSV(df, subName):
    df.to_csv(f'./result-{subName}.csv')

def download_stockDataset(con, marketName):
    # define Bloomberg API con first in Notebook
    folder_name = f"downloads-{marketName}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    bd_columns = ['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST', 'VOLUME', 'CUR_MKT_CAP', 'CHG_PCT_1D']
    start_date = '20150101'
    end_date = '20230724'
    df_members = pd.read_csv(f'dataset-{marketName}-members.csv')
    tickers = df_members['value']
    for ticker in tqdm(tickers):
        try:
            df = con.bdh(f'{ticker} Equity', bd_columns, start_date, end_date)
            columns = [column for (ticker,column) in df.columns]
            df.columns = columns
            tickerName = ticker.replace('/', '_')
            df.to_csv(f'./{folder_name}/dataset-{marketName}-{tickerName}.csv')
        except Exception as err:
            print(ticker)

def openCorrmatrixFile(ticker_target, ticker_source, lag, max_period, max_shift, stride, folder_name):
    subfolder_name = f'results-corrmatrix-{ticker_target}-pairs-lag{lag}-p{max_period}-d{max_shift}-s{stride}'
    folder_path = f'./{folder_name}/{subfolder_name}/'
    file_name = folder_path+f'corrmatrix-lag{lag}-{ticker_target}-{ticker_source}.csv'
    df = openCSV(file_name)    
    return df

def plot_heatmap(df):
    plt.figure(figsize=(30, 10))  # 그래프의 크기 설정
    ax = sns.heatmap(df, vmin=-1, vmax=1, cmap='coolwarm', annot=False, fmt=".2f")
    ax.set_title(f'Correlation heatmap: lagged source( ) -> target( ) ...')
    ax.set_xlabel('shift')
    ax.set_ylabel('period')
    plt.show()

def getCorrMean(df):
    corr_matrix = df
    df = corr_matrix.mean().to_frame('corr_mean')
    df['corr_mean_MA(20)'] = df['corr_mean'].rolling(window=20, min_periods=1).mean()
    df.index = pd.to_datetime(df.index)
    return df

def getCorrStat(df):
    corr_stat = getCorrMean(df).describe()
    return corr_stat

def getCorrStat2(df, cutoff=0.05):
    df = getCorrMean(df)
    mean_global = round(df['corr_mean_MA(20)'].mean(), 4)
    num_global = len(df)
    df_non_zeros = df[df['corr_mean_MA(20)']>cutoff]['corr_mean_MA(20)']
    mean_non_zeros = round(df_non_zeros.mean(), 4)
    mean_non_zeros = mean_non_zeros if not math.isnan(mean_non_zeros) else 0
    num_non_zeros = len(df_non_zeros)
    df_zeros = df[df['corr_mean_MA(20)']<=cutoff]['corr_mean_MA(20)']
    mean_zeros = round(df_zeros.mean(), 4)
    num_zeros = len(df_zeros)
    ratio_non_zeros = num_non_zeros / num_global
    corr_stat2 = {'mean_global': mean_global, 'num_global': num_global, 'mean_non_zeros': mean_non_zeros, 'num_non_zeros': num_non_zeros, 'mean_zeros': mean_zeros, 'num_zeros': num_zeros, 'ratio_non_zeros': ratio_non_zeros}
    return corr_stat2

def getTimelagScore(df, ticker_target, ticker_source, lag, option=None):
    corr_stat2 = getCorrStat2(df)
    quality = int(round(corr_stat2['mean_global'], 2)*100)
    strength = int(round(corr_stat2['mean_non_zeros'], 2)*100)
    prevalence = int(round(corr_stat2['ratio_non_zeros'], 2)*100)
    total_score = round(np.mean([quality, strength, prevalence]), 2)

    if option == 'deploy':
        # quality_correction = 100 if quality*2 >= 100 else quality*2
        # strength_correction = 100 if strength*2 >= 100 else strength*2  
        volatility = int(round(getCorrStat(df)['corr_mean'].loc['std'], 2)*100)
        # total_score_correction = round(np.mean([quality_correction, strength_correction, prevalence]), 2)
        score = {'target': ticker_target, 'source':ticker_source, 'lag':lag, 'quality': quality, 'strength': strength, 'volatility': volatility, 'prevalence': prevalence, 'total_score':total_score}
    else: 
        score = {'target': ticker_target, 'source':ticker_source, 'lag': lag, 'quality': quality, 'strength': strength, 'prevalence': prevalence, 'total_score':total_score}

    return score

def plot_periodicCorr(df):
    df = getCorrMean(df)
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(df.index, df['corr_mean'], label='Original', linewidth=1)
    ax.plot(df.index, df['corr_mean_MA(20)'], label='Moving Average (window=20)', linewidth=2)

    ax.set_xlabel('Date')
    ax.set_ylabel('corr_mean')
    ax.set_title(f'Periodic correlation by shift: ')
    ax.legend()

    # x축 틱 설정
    num_ticks = 20  
    total_rows = len(df)
    stride = max(total_rows // (num_ticks - 1), 1)
    xticks = df.index[::stride].tolist()[:-1]
    xticks.append(df.index[-1])  # 마지막 날짜를 추가
    ax.set_xticks(xticks)

    plt.xticks(rotation=45)

    # 오른쪽 값에 해당하는 위치에 값을 표시하고 가로 점선 그리기
    right_value = df['corr_mean_MA(20)'].iloc[-1]
    ax.axhline(right_value, color='black', linestyle='--')
    ax.annotate(f'{right_value:.2f}', xy=(df.index[-1], right_value),
                xytext=(20, 5), textcoords='offset points', color='black', ha='right')
    # q2_value = round(self.getCorrStat().loc['50%']['corr_mean'], 2)
    # ax.axhline(y=q2_value, color='gray', linestyle='--')
    # ax.annotate(f'Q2={q2_value}', xy=(df.index[0], q2_value), xytext=(20, 0), textcoords='offset points', color='gray', ha='left', va='center')

    plt.grid(axis='x')
    plt.show()        

def getQuickPair(ticker_target, ticker_source):
    return Pair(K200stock(ticker_target), R3000stock(ticker_source))

def run(intermarket, lag, min_period, max_period, max_shift, stride, rng, task_memo=None):
    time = record_time()
    time2 = record_time(option='min')
    config = f'lag{lag}_period{min_period}-{max_period}_days{max_shift}_stride{stride}'
    if task_memo == None:
        task_memo = f'{time}-{config}'
    
    dict_config = {
        'start_time': time2,
        'lag': lag,
        'min_period': min_period,
        'max_period': max_period,
        'max_shift': max_shift,
    'stride': stride
    }
    pd.DataFrame([dict_config]).to_csv(f'task-config-{time2}.csv')

    targets = intermarket.target_market.members[rng[0]:rng[1]] if rng != None else intermarket.target_market.members
    num_of_targets = len(targets)

    errs = []
    for i in range(num_of_targets):
        try:
            intermarket.analyze_corrmatrices_parallel(task_memo=task_memo, save_time=False, target_index=i, lag=lag, min_period=min_period, max_period=max_period, max_shift=max_shift, stride=stride)
        except Exception as err:
            errs.append({'index': i, 'error':err})
    
    pd.DataFrame(errs).to_csv(f'task-errs-{time2}.csv')

def set_config(lag, min_period, max_period, max_shift, stride, memo=None):
    config = {'lag':lag, 'min_period':min_period, 'max_period': max_period, 'max_shift': max_shift, 'stride': stride, 'memo': memo, 'config_info': f'lag{lag}-p{min_period}_{max_period}-d{max_shift}-s{stride}'}
    return config

def analyze_intermarketTimelag(target_market, source_market, config, lag=0, save=True):
    start_time = time.time()
    print(f"start: analyze intermarket ...")

    lag = config['lag']
    min_period = config['min_period']
    max_period = config['max_period']
    max_shift = config['max_shift']
    stride = config['stride']
    folder_name = config['memo']
    config_info = config['config_info']
    results = []
    for ticker_target in tqdm(target_market.tickers):
        for ticker_source in source_market.tickers:
            print(f'pair: {ticker_target} x {ticker_source}')
            if ticker_target != ticker_source:
                try:
                    df = openCorrmatrixFile(ticker_target, ticker_source, lag, max_period, max_shift, stride, folder_name=folder_name)
                    score = getTimelagScore(df, ticker_target, ticker_source, lag)
                    
                    if ticker_target != None and 'KS' in ticker_target:
                        target = K200stock(ticker_target)
                    elif ticker_target != None and 'U' in ticker_target:
                        target = R3000stock(ticker_target)
                    if ticker_source != None and 'KS' in ticker_source:
                        source = K200stock(ticker_source)
                    elif ticker_source != None and 'U' in ticker_source:
                        source = R3000stock(ticker_source)
                    score['GICS_SECTOR_NAME_target'] = target.getInfo()['GICS_SECTOR_NAME']
                    score['GICS_SECTOR_NAME_source'] = source.getInfo()['GICS_SECTOR_NAME']

                    results.append(score)

                except Exception as err:
                    score = {'target': ticker_target, 'source':ticker_source, 'lag':lag}
                    results.append(score)
    df = pd.DataFrame(results)
    df = df.sort_values(by='total_score', ascending=False)
    df = df.reset_index(drop=True)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"end: {round(execution_time, 2)} seconds elapsed")

    if save == True:
        df.to_csv(f'result-timelag-analysis-{config_info}.csv', index=False)
    
    return df

def analyze_intermarketTimelag_parallel(target_market, source_market, config, lag=0, save=True):
    start_time = time.time()
    print(f"start: analyze intermarket ...")

    lag = config['lag']
    min_period = config['min_period']
    max_period = config['max_period']
    max_shift = config['max_shift']
    stride = config['stride']
    folder_name = config['memo']
    config_info = config['config_info']

    def f(ticker_target):
        print(f'pair: {ticker_target} x {ticker_source}')
        for ticker_source in source_market.tickers:
            if ticker_target != ticker_source:
                try:
                    df = openCorrmatrixFile(ticker_target, ticker_source, lag, max_period, max_shift, stride, folder_name=folder_name)
                    score = getTimelagScore(df, ticker_target, ticker_source, lag)
                    
                    if ticker_target != None and 'KS' in ticker_target:
                        target = K200stock(ticker_target)
                    elif ticker_target != None and 'U' in ticker_target:
                        target = R3000stock(ticker_target)
                    if ticker_source != None and 'KS' in ticker_source:
                        source = K200stock(ticker_source)
                    elif ticker_source != None and 'U' in ticker_source:
                        source = R3000stock(ticker_source)
                    score['GICS_SECTOR_NAME_target'] = target.getInfo()['GICS_SECTOR_NAME']
                    score['GICS_SECTOR_NAME_source'] = source.getInfo()['GICS_SECTOR_NAME']
                except Exception as err:
                    score = {'target': ticker_target, 'source':ticker_source, 'lag':lag}
        return score

    with Parallel(n_jobs=mp.cpu_count()) as parallel:
        results = parallel(delayed(f)(ticker_target) for ticker_target in target_market.members)

    df = pd.DataFrame(results)
    df = df.sort_values(by='total_score', ascending=False)
    df = df.reset_index(drop=True)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"end: {round(execution_time, 2)} seconds elapsed")

    if save == True:
        df.to_csv(f'result-timelag-analysis-{config_info}.csv', index=False)
    
    return df    

def getPairFromTimelagAnalysis(df, i, stock=None):
    row = df.iloc[i]
    if stock == None:
        ticker_target = row['target']
        if ' KS' in ticker_target:
            stock = K200stock(ticker_target)
        elif ' U' in ticker_target:
            stock = R3000stock(ticker_target)    
    ticker_source = row['source']
    if ticker_source != None and ' KS' in ticker_source:
        source = K200stock(ticker_source)
    elif ticker_source != None and ' U' in ticker_source:
        source = R3000stock(ticker_source)
    p = Pair(stock, source)
    return p