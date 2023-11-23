# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 20:08:13 2023

@author: User
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
import scipy.stats
import statannot

data = pd.read_csv('survey lung cancer.csv')
### Info about data
print('Data Shape:', data.shape)
print('Data summary:', data.info)
### Check missing data
sns.heatmap(data.isna().transpose(),cmap=sns.color_palette("hls", 6))
print('missing data:',data.isnull().sum())
### Check/drop duplicated data
print('duplicated data number:',data.duplicated().sum())
print('duplicated data index:',data.index[(data.duplicated())])
data_dup = data.copy()
data_dup.drop_duplicates(inplace=True)
print('Data Shape(drop dup):', data_dup.shape)
data_dup_feature_name = data.columns[0:15]
data_dup_dis_name = data.columns[15]
### 轉換文字為數字，Label encoder
encoder = LabelEncoder()
data_dup['LUNG_CANCER'] = encoder.fit_transform(data_dup['LUNG_CANCER'])
data_dup['GENDER'] = encoder.fit_transform(data_dup['GENDER'])
### 畫出Age的分布(hist, kde, bar plot)
fig, ax = plt.subplots(1,3,figsize=(20,6)) ###figsize是先寬再長
sns.histplot(data_dup['AGE'],ax=ax[0]) ###參數 hist是柱狀, kde是曲線 bin是區間數
sns.kdeplot(data = data_dup,x='AGE',ax=ax[1],hue='LUNG_CANCER',fill=True)
sns.boxplot(x=data_dup['LUNG_CANCER'],y=data_dup['AGE'],ax=ax[2])
plt.suptitle("Visualizing AGE column",size=20)
### 計算independant t test，看兩組間的age是否有差異
age_tt = scipy.stats.ttest_ind(
    data_dup[(data_dup['LUNG_CANCER'] == 0)]['AGE'],
    data_dup[(data_dup['LUNG_CANCER'] == 1)]['AGE']
    )
plt.show()
### 區分類別和連續性變項
data_categ_col = []
data_cont_col = ['AGE']
for i in data_dup.columns:
    if i != 'AGE':
        data_categ_col.append(i)
### 畫出類別變項在lung Cancer和 Normal間的差異
fig,ax = plt.subplots(15,2,figsize=(30,90))
for index,i in enumerate(data_categ_col):   ###enumerate 會遍歷每個值並回傳index
    sns.countplot(data=data_dup,x=i,ax=ax[index,0])  ### 先畫total，下面hue再畫分lungCa
    sns.countplot(data=data_dup,x=i,ax=ax[index,1],hue='LUNG_CANCER')
fig.tight_layout()  ### 做子圖時為了避免標籤互相干擾，所以要分開來給個距離
fig.subplots_adjust(top=0.95)   ### 調整上面的空間
plt.suptitle("Visualizing Categorical Columns",fontsize=50)
### 用熱點圖呈現每個變項之間的關係係數
### 要注意.corr()預設是pearson，適用於連續性變項，所以類別變項要用spearman
plt.figure(figsize=(15,15))
sns.heatmap(data_dup.corr(method='spearman'),annot=True,linewidth=0.5,fmt='0.2f')    ###annot是會在每個格子中加上數字，fmt代表會取到小數點第二位


