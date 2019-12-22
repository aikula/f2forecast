from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import ThresholdAD
from adtk.transformer import NaiveSeasonalDecomposition
from adtk.detector import OutlierDetector
from sklearn.neighbors import LocalOutlierFactor

import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse

from statsmodels.tsa.api import ExponentialSmoothing

import statsmodels.api as sm
import itertools

import warnings
warnings.filterwarnings("ignore") # specify to ignore warning messages


# функция графика сезонности
def sesonal(data, s):
    plt.figure(figsize=(19,8), dpi= 80)
    for i, y in enumerate(data.index.year.unique()):
        plt.plot(list(range(1,s+1)), data[data.index.year==y][data.columns[0]].values, label=y)
    plt.title("Сезонность по периодам")
    plt.legend(loc="best")
    plt.show()
    
from statsmodels.tsa.stattools import adfuller
#На вход датафрейм, обязательно со столбцом Y
def metrics(real, forecast):
    
    print("Тест на стационарность:")
    dftest = adfuller(real[real.columns[0]]-forecast, autolag='AIC')
    print("\tT-статистика = {:.3f}".format(dftest[0]))
    print("\tP-значение = {:.3f}".format(dftest[1]))
    print("Критические значения :")
    for k, v in dftest[4].items():
        print("\t{}: {} - Данные {} стационарны с вероятностью {}% процентов".format(k, v, "не" if v<dftest[0] else "", 100-int(k[:-1])))
    
    real=np.array(real[real.columns[0]].values)
    forecast=np.array(forecast)
    print('MAD:', round(abs(real-forecast).mean(),4))
    print('MSE:', round(((real-forecast)**2).mean(),4))
    print('MAPE:', round((abs(real-forecast)/real).mean(),4))
    print('MPE:', round(((real-forecast)/real).mean(),4))
    print('Стандартная ошибка:', round(((real-forecast)**2).mean()**0.5,4)) 
    
def auto_sarima(data, s):
    # Define the p, d and q parameters to take any value between 0 and 2
    p = d = q = range(0, 2)

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], s) for x in list(itertools.product(p, d, q))]
    
    best=[]
    
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(data,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

                results = mod.fit()
                
                p_val_status='True'
                for pv in results.pvalues:
                    if pv>0.05:
                        p_val_status='False'
                        break
                if len(best)==0 and p_val_status=="True":
                    best=[param, param_seasonal, results.aic, results.pvalues]
                if best[2]>results.aic and p_val_status=="True":
                    best=[param, param_seasonal, results.aic, results.pvalues]

                print('ARIMA{}x{} - AIC:{} - P-values: {}'.format(param, param_seasonal, round(results.aic,4), p_val_status))
            except:
                continue
    print('Лучшая модель:')
    print('ARIMA{}x{} - AIC:{} - P-values: {}'.format(best[0], best[1], round(best[2],4), best[3]))