"""Later review implementation; not historical employer code."""
from pathlib import Path
import argparse
import ast
import json
import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def rolling_beta(stock, proxy, window=390, minimum=200):
    numerator = stock.mul(proxy, axis=0).rolling(window, min_periods=minimum).sum().shift(1)
    denominator = proxy.pow(2).rolling(window, min_periods=minimum).sum().shift(1)
    return numerator.div(denominator.replace(0, np.nan), axis=0)

def qtc_demo():
    rng=np.random.default_rng(7);p=pd.Series(rng.normal(0,.001,900))
    s=pd.DataFrame({'A':1.5*p,'B':.7*p})
    beta=rolling_beta(s,p)
    assert np.allclose(beta.A.dropna(),1.5)
    changed=s.copy();changed.loc[500,'A']+=.2
    assert rolling_beta(changed,p).loc[500,'A']==beta.loc[500,'A']
    assert rolling_beta(changed,p).loc[501,'A']!=beta.loc[501,'A']
    noisy=s+rng.normal(0,.0003,s.shape)
    residual=noisy-rolling_beta(noisy,p).mul(p,axis=0)
    accumulated=residual.rolling(30,min_periods=30).sum()
    mean=accumulated.rolling(390,min_periods=390).mean().shift(1)
    std=accumulated.rolling(390,min_periods=390).std().shift(1)
    z=(accumulated-mean)/std.replace(0,np.nan)
    return {'synthetic':True,'beta_known_value_test':True,'beta_current_bar_not_used':True,
            'first_full_z_row':int(z.dropna().index[0]),
            'scope':'Signal and warm-up mechanics only; not original QTC PnL replication'}
if __name__ == '__main__':
    print(json.dumps(qtc_demo(),indent=2))
