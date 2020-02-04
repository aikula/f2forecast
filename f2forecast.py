import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
#import time

from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import ThresholdAD
from adtk.detector import OutlierDetector

from sklearn.neighbors import LocalOutlierFactor
#from sklearn.metrics import r2_score

from scipy.stats import variation

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
from statsmodels.stats.outliers_influence import variance_inflation_factor

#from dateutil.parser import parse

#import itertools
from itertools import compress, product

#import pmdarima as pm
from pmdarima import auto_arima

import warnings
warnings.filterwarnings("ignore") # specify to ignore warning messages

# функция графика сезонности
def sesonal(data, s):
    plt.figure(figsize=(19,8), dpi= 80)
    for i, y in enumerate(data.index.year.unique()):
        plt.plot(list(range(1,len(data[data.index.year==y])+1)), data[data.index.year==y][data.columns[0]].values, label=y)
    plt.title("Сезонность по периодам")
    plt.legend(loc="best")
    plt.show()
    
def metrics(real, forecast):
    
    if type(real)==pd.core.frame.DataFrame:
        real=real[real.columns[0]].values
    
    print("Тест на стационарность:")
    dftest = adfuller(real-forecast, autolag='AIC')
    print("\tT-статистика = {:.3f}".format(dftest[0]))
    print("\tP-значение = {:.3f}".format(dftest[1]))
    print("Критические значения :")
    for k, v in dftest[4].items():
        print("\t{}: {} - Данные {} стационарны с вероятностью {}% процентов".format(k, v, "не" if v<dftest[0] else "", 100-int(k[:-1])))
    
    #real=np.array(real[real.columns[0]].values)
    forecast=np.array(forecast)
    print('MAD:', round(abs(real-forecast).mean(),4))
    print('MSE:', round(((real-forecast)**2).mean(),4))
    print('MAPE:', round((abs(real-forecast)/real).mean(),4))
    print('MPE:', round(((real-forecast)/real).mean(),4))
    print('Стандартная ошибка:', round(((real-forecast)**2).mean()**0.5,4)) 
    

def metrics_short(real, forecast):
    real=np.array(real[real.columns[0]].values)
    forecast=np.array(forecast)
    print('MAD:', round(abs(real-forecast).mean(),4))
    print('MSE:', round(((real-forecast)**2).mean(),4))
    print('MAPE:', round((abs(real-forecast)/real).mean(),4))
    print('MPE:', round(((real-forecast)/real).mean(),4))
    print('Стандартная ошибка:', round(((real-forecast)**2).mean()**0.5,4)) 
    
def h_map(data, level):
    corr = data.corr()
    plt.figure(figsize=(14, 14))
    sns.heatmap(corr[(corr >= level) | (corr <= -level)],
            cmap="RdBu_r", vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True)
    plt.show()
    
#небольшая функция для построения набора комбинаций переменных
def combinations(items):
    return list( set(compress(items,mask)) for mask in product(*[[0,1]]*len(items)) )

def get_factors(data, Y, columns):

    # колонки, которые показали свою значимость в процессе отбора критериев
    # переменная spisCol хранит варианты комбинаций все переменных
    spisCol=combinations(columns)

    print('Количество комбинаций ', len(spisCol))
    
    #добавим константу в набор данных, нужна для рассчета регрессии
    data=sm.add_constant(data)

    #сохраним в этом списке данные лучших моделей
    arr_res=[]

    #пробежимся циклом по всем вариантам комбинаций
    for c in spisCol:
        perem=list(c)
        flag=True
    
        if len(perem)==0: continue
        
        if not('const' in c):
            perem.append('const')
        
        # если больше одного клитерия, рассчитаем VIF    
        if len(perem)>1:
            vif = [variance_inflation_factor(data[perem].values, i) for i in range(data[perem].shape[1])]
        else:
            vif=[]
    
        #проверим список VIF, если хоть одна переменная больше 1000 (очень большое значение, на самом деле),
        #то в модели присутсвует мультиколлинераность
        for vv in vif:
            if vv>1000: 
                flag=False
        
        #посчитаем саму модель
        reg = sm.OLS(Y, data[perem])
        res=reg.fit()

        #отбросим нулевую гипотезу для всех регрессоров конкретной модели
        for val in res.tvalues:
            if val<2 and val>-2:
                flag=False
                break
        for val in res.pvalues:
            if val>0.05:
                flag=False
                break
        #если нулевую гипотезу отбросили и VIF в норме, сохраним результаты
        if flag:
            re=np.array(res.fittedvalues.copy())
            MSE=((np.array(Y)-re)**2).sum()/len(re)
            
            MAPE=(abs((np.array(Y)-re)/np.array(Y))).sum()/len(re)
        
            arr_res.append([round(MSE,4), res.rsquared, perem])

    #отсортируем и выведем результаты
    arr_res.sort()
    df_model=pd.DataFrame(arr_res, columns=['MSE', 'r2', 'Переменные'])
    print('Результаты перебора в порядке возрастания MSE:')
    print(df_model)
    return df_model