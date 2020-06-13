import numpy as np
import pandas as pd
import scipy.stats as st
from plotnine import *
import seaborn as sns
from sklearn.linear_model import LinearRegression
import jenkspy

#from https://stats.stackexchange.com/questions/143974/jenks-natural-breaks-in-python-how-to-find-the-optimum-number-of-breaks
def goodness_of_variance_fit(array, classes):
    # get the break points
    classes = jenkspy.jenks_breaks(array, classes)

    # do the actual classification
    classified = np.array([classify_jenks(i, classes) for i in array])

    # max value of zones
    maxz = max(classified)

    # nested list of zone indices
    zone_indices = [[idx for idx, val in enumerate(classified) if zone + 1 == val] for zone in range(maxz)]

    # sum of squared deviations from array mean
    sdam = np.sum((array - array.mean()) ** 2)

    # sorted polygon stats
    array_sort = [np.array([array[index] for index in zone]) for zone in zone_indices]

    # sum of squared deviations of class means
    sdcm = sum([np.sum((classified - classified.mean()) ** 2) for classified in array_sort])

    # goodness of variance fit
    gvf = (sdam - sdcm) / sdam

    return gvf

def classify_jenks(value, breaks):
    for i in range(1, len(breaks)):
        if value < breaks[i]:
            return i
    return len(breaks) - 1

def calc_qq(df,col):
    sample_qs = [(np.percentile(df[col],i)-np.mean(df[col]))/np.std(df[col]) for i in range(5,100,5)]
    theoretical_qs = [st.norm.ppf(i/100) for i in range(5,100,5)]
    qq = pd.DataFrame({'sample_q':sample_qs,'theoretical_q':theoretical_qs})
    reg = LinearRegression(fit_intercept=False).fit(np.array(qq['theoretical_q'])[:,None], 
                                 np.array(qq['sample_q'])[:,None])
    return qq, reg

def boxcox(ser,lamb=0):
    ser+= 1 - ser.min()
    if lamb==0: 
        return np.log(ser)
    else:
        return (ser**lamb - 1)/lamb
    
def boxcox_lamb_df(ser, ls = [i/10 for i in range(-30,31,5)]):
    coefs = []
    for l in ls:
        df = pd.DataFrame.from_dict({'val': boxcox(ser,l)})
        qq, reg = calc_qq(df,'val')
        coefs.append(reg.coef_.squeeze().item())
    return pd.DataFrame({'lamb':ls,'coef':coefs})

def boxcox_lamb(ser, ls = [i/10 for i in range(-30,31,5)]):
    df = boxcox_lamb_df(ser,ls)
    return df.lamb[df.coef.idxmax()]

def boxcox_plot(df, col, ls = [i/10 for i in range(-30,31,5)]):
    lamb_df = boxcox_lamb_df(df[col],ls)
    g = (ggplot(lamb_df, aes(x='lamb',y='coef',group=1)) + 
         geom_point() + geom_line())
    return g