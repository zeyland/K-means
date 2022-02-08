#################################
#Customer Segmentation with K-Means
#################################

# IS PROBLEMI
# Kural tabanlı müşteri segmentasyonu  yöntemi RFM ile
# makine öğrenmesi yöntemi  olan K-Means'in müşteri segmentasyonu için
# karşılaştırılması beklenmektedir.

# VERI SETI
# Online Retail II isimli veri seti İngiltere merkezli online bir satış
# mağazasının 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını
# içermektedir

# DEGISKENLER
# InvoiceNO – Fatura Numarası
# Eğer bu kod C ile başlıyorsa işlemin iptal edildiğini ifade eder.
# StockCode – Ürün Kodu
# Her bir ürün için eşsiz numara
# Description – Ürün İsmi
# Quantity – Ürün Adedi
# Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate – Fatura tarihi
# UnitPrice – Fatura fiyatı (Sterlin)
# CustomerID – Eşsiz müşteri numarası
# Country – Ülke ismi

#################################################

import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from helpers.tyb import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_excel("Dataset/online_retail_II.xlsx")
df = df_.copy()
df.head()
df.isnull().sum()
df.shape
df.describe().T

###############################################################
# Veri Hazırlama (Data Preparation)
###############################################################
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[(df['Quantity'] > 0)]
df = df[(df['Price'] > 0)]
df["TotalPrice"] = df["Quantity"] * df["Price"]

cat_cols, num_cols,cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in ["InvoiceDate",  'Customer ID']]

check_df(df)
# %99 dan sonra aykırılıkları gözüküyor.
len(df[df["Price"] > df['Price'].quantile(0.99)])
len(df[df["Quantity"] > df['Quantity'].quantile(0.99)])
# %99 dan fazla yaklaşık 3300 kişi var, verinin yüzde 0.8 sine denk geliyor.
# genele bir sınıflandırma yapmak istediğim ve genelden çok aykırı olduğu için direkt baskılıyorum.
for col in num_cols:
    replace_with_thresholds(df, col, 0.01,  0.99)


###############################################################
# RFM Metriklerinin Hesaplanması (Calculating RFM Metrics)
###############################################################

df["InvoiceDate"].max()
today_date = dt.datetime(2010, 12, 11)

rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                     'Invoice': lambda Invoice: Invoice.nunique(),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

rfm.columns = [ 'recency', 'frequency', 'monetary' ]
rfm.head()

check_df(rfm)
for col in rfm:
    replace_with_thresholds(rfm, col, 0.01,  0.99)

########################
# K-means Standartlaştırma
########################
df.head()
df = rfm
sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)

df[0:5]

################################
# Optimum Küme Sayısının Belirlenmesi
################################

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 15))
elbow.fit(df)
elbow.show()

elbow.elbow_value_

###########
#K-Means
###########

kmeans = KMeans(n_clusters=elbow.elbow_value_)
k_fit = kmeans.fit(df)

k_fit.n_clusters
k_fit.cluster_centers_
k_fit.labels_
k_fit.inertia_

df[0:5]

################################
# Kümelerin Görselleştirilmesi
################################

k_means = KMeans(n_clusters=elbow.elbow_value_).fit(df)
kumeler = k_means.labels_
type(df)
df = pd.DataFrame(df)

plt.scatter(df.iloc[:, 0],
            df.iloc[:, 1],
            c=kumeler,
            s=50,
            cmap="viridis")
plt.show()

# merkezlerin isaretlenmesi
merkezler = k_means.cluster_centers_

plt.scatter(df.iloc[:, 0],
            df.iloc[:, 1],
            c=kumeler,
            s=50,
            cmap="viridis")

plt.scatter(merkezler[:, 0],
            merkezler[:, 1],
            c="red",
            s=200,
            alpha=0.8)
plt.show()


################################
# Final Cluster'ların Oluşturulması
################################

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)
kumeler = kmeans.labels_

df = rfm
df.head()

pd.DataFrame({"Customer ID ": df.index, "Kumeler": kumeler})

df["cluster_no"] = kumeler

df["cluster_no"] = df["cluster_no"] + 1

df.head()

df.groupby("cluster_no").agg({"cluster_no": "count"})
df.groupby("cluster_no").agg(np.mean)
