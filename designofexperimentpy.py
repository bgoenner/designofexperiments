# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 17:30:35 2018

@author: bg
"""

import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# Common Functions file for DoE

def splitDataFrame(dataFrame, column):
    
    DataframeArray = [pd.DataFrame] * dataFrame[column].unique().size
    i = 0

    #creats and array
    for c in dataFrame[column].unique():
        DataframeArray[i] = dataFrame[dataFrame[column] == c].reset_index(drop=True)
        i = i + 1

    i = None
        
    return DataframeArray

def compareGroups(dataFrame, Gcolumn, Dcolumn):
    numberGroups = dataFrame[Gcolumn].unique().size
    probDataFrame = pd.DataFrame(data=np.empty([numberGroups, numberGroups]),
                                 index= dataFrame[Gcolumn].unique(),
                                 columns= dataFrame[Gcolumn].unique())

    dataList = splitDataFrame(dataFrame, Gcolumn)


    for group1 in dataList:
        for group2 in dataList:
            # z-test for group
            group1_d = group1[Dcolumn].as_matrix()
            group2_d = group2[Dcolumn].as_matrix()
            if group1 is not group2:
                # print(group1[Dcolumn].size, group1.index.size, group2[Dcolumn].size, group2.index.size)
                probDataFrame.loc[group1[Gcolumn][0], group2[Gcolumn][0]] = \
                    ztest(group1_d, group2_d, alternative='two-sided')[1]
            else:   
                probDataFrame.loc[group1[Gcolumn][0], \
                              group2[Gcolumn][0]] = float('nan')
                
    return probDataFrame

def compareToNormal(dataFrame, title=None):
    normal = np.random.standard_normal(size=dataFrame.size)
    data = dataFrame.sort_values()    
    normal.sort()
    plt.figure()
    plt.scatter(normal, data)
    
    # best fit line
    plt.plot(normal, np.poly1d(np.polyfit(normal, data, 1))(normal))
    #plt.plot(normal.sort(), normal.sort())
    if title != None:
        plt.title(title + ' qunatile-quantile plot')
        plt.xlabel('Normal distribution')
        plt.ylabel('Data')

    plt.show()
    
    
def histogramDOE(dataFrame, title=None, Xaxis='data'):
    # the histogram of the data
    n, bins, patches = plt.hist(dataFrame, 50, normed=1, facecolor='green', alpha=0.75)
    
    # add a 'best fit' line
    mu = dataFrame.mean()
    sigma = dataFrame.std()
    
    y = mlab.normpdf( bins, mu, sigma)
    plotHist = plt.plot(bins, y, 'r--', linewidth=1)
    
    plt.xlabel(Xaxis)
    plt.ylabel('Probability')
    plt.title(r'$\mathrm{Histogram\ of\ ' + title + ' :}\ \mu=' + "{:10.3f}".format(mu) + ',\ \sigma=' + "{:10.3f}".format(sigma) +'$')
    #plt.axis([BrandA.loc[:, 'charge.temp'].min() - 5, BrandA.loc[:, 'charge.temp'].max() + 5, 0, 0.03])
    plt.grid(True)
    
    plt.show()
    
def boxPlotData(dataFrame, columnGroup, columnData, title='Data'):
    data = [pd.DataFrame()] * len(dataFrame[columnGroup].unique())

    i = 0
    
    SdataFrame = splitDataFrame(dataFrame, columnGroup)
    
    for element in SdataFrame:
        data[i] = element[columnData]
        i = i+1
    
    i = None
    
    fig = plt.figure()
    plt.title(title)
    plt.boxplot(data)
    plt.show()    
    