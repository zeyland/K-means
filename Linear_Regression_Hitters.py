# Hiter Prediction with Linear Regression
######################################################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action="ignore")
pd.set_option('display.float_format', lambda x: '%.2f' % x)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv("Dataset/hitters.csv")
df.shape
df.head()
df.nunique()
df.describe().T
df["Salary"].mean()

# Aykırı gözlem

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)

def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
check_outlier(df, num_cols)
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
for col in num_cols:
    replace_with_thresholds(df, col)

# Feature Engineering
# isabet sayısı / yapılan vuruş sayısı
df["Hits_AtBat"]= df["Hits"]/df["AtBat"]

# en degerlı vurus sayısı / yapılan vurus sayısı
df["HmRun_AtBat"]= df["HmRun"]/df["AtBat"]

# karsı oyuncuya hata yaptırma oranı
df["Walks_CWalks"]= df["Walks"]/df["CWalks"]

# oyuncunun takımına sayı kazandırma oranı
df["Runs_CRuns"]= df["Runs"]/df["CRuns"]

## en degerlı vurus sayısının oranı
df["HmRun/CHmRun"]= df["HmRun"]/df["CHmRun"]

# ısabet sayısı
df["Hits_CHits"]= df["Hits"]/df["CHits"]

# kosu yaptırma oranı
df["RBI_CRBI"]= df["RBI"]/df["CRBI"]

# oyuncunun topa vurma oranı
df["AtBat_CAtBat"]= df["AtBat"]/df["CAtBat"]
df.head()

df.isnull().values.any()
df = df.dropna()
df.shape

# kategorik değişkenlerin analizi
cat_cols

df.groupby("NewLeague").agg({"Salary": "mean"})\
    .sort_values("NewLeague", ascending=False).head(10)

#division ordinary cat. - label encoding
#'League', 'NewLeague cat but not ordinal - one hot
cat_cols_label =[col for col in cat_cols if col not in ["Division"]]

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
for col in cat_cols_label:
    df = one_hot_encoder(df, [col], drop_first=True)

#for division, label encoding -binary encoding-
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
df = label_encoder(df, "Division")

## Salary deki boşluklar için knn'in uygulanması.
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
df.head()

#Linear Regression
def mse(actual, predicted):
    return np.mean((actual - predicted) ** 2)


def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))


def mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))

X = df.drop('Salary', axis=1)
y = df[["Salary"]]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=1)
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# sabit (b - bias)
reg_model.intercept_[0]

# coefficients (w - weights)
reg_model.coef_


##########################
# Tahmin Başarısını Değerlendirme/ HOLDOUT
##########################

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
# rmse 230.3594734528375

# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# 320.46278682562433 test verisi üzerinde hatamız daha fazla.

# TRAIN RKARE
reg_model.score(X_train, y_train)

# Test RKARE
reg_model.score(X_test, y_test)


# Test Model görselleştirilmesi:

g = sns.regplot(x=y_test, y=y_pred, scatter_kws={'color': 'b', 's': 5},
                ci=False, color="r")
g.set_title(f"Test Model R2: = {reg_model.score(X_test, y_test):.4f}")
g.set_ylabel("Predicted Salary")
g.set_xlabel("Salary")
plt.xlim(-5, 2700)
plt.ylim(bottom=0)
plt.show()

# 10 Katlı CV-Cross Validation- RMSE
np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=10, scoring="neg_mean_squared_error")))
#312.2650690285747

######################################################
# Simple Linear Regression with Gradient Descent from Scratch/ gözlük iki
#b , w değerlerini mse yi min yapacak şekilde optimize etmek
#cost function mse yi hesaplarken, update_weight fonksiyonu w,b değerlerini optimal ediyor.
#train fonksiyonu ikisini de birleştiriyor.
######################################################
def cost_function(Y, b, w, X):
    m = len(Y)  # gözlem sayısı
    sse = 0  # toplam hata
    # butun gozlem birimlerini gez:
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2
    mse = sse / m
    return mse

# update_weights
def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)

    b_deriv_sum = 0
    w_deriv_sum = 0

    for i in range(0, m):

        y_hat = b + w * X[i]

        y = Y[i]

        b_deriv_sum += (y_hat - y)

        w_deriv_sum += (y_hat - y) * X[i]

    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w

# train fonksiyonu
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                   cost_function(Y, initial_b, initial_w, X)))

    b = initial_b
    w = initial_w
    cost_history = []
    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)

        if i % 100 == 0:
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse))

    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w

#Let's say
dff = df
X = df["CHits"]
Y = df["Salary"]

# hyperparameters
learning_rate = 0.001
initial_b = 699.478
initial_w = 7.30
num_iters = 1000

train(Y, initial_b, initial_w, X, learning_rate, num_iters)