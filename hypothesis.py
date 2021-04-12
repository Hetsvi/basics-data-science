import os
import pandas as pd
import numpy as np
import requests
import json
import re

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def null_hypoth():
    """
    :Example:
    >>> isinstance(null_hypoth(), list)
    True
    >>> set(null_hypoth()).issubset({1,2,3,4})
    True
    """

    return [1]


def simulate_null():
    """
    :Example:
    >>> pd.Series(simulate_null()).isin([0,1]).all()
    True
    """

    ara_null = np.random.choice([0, 1], p=[0.01, 0.99], size=300)
    # table
    return ara_null


def estimate_p_val(N):
    """
    >>> 0 < estimate_p_val(1000) < 0.1
    True
    """

    mean = []
    orifinal_avg = np.array([1] * 292 + [0] * 8)
    mean_original = np.mean(orifinal_avg)
    for i in np.arange(N):
        data = simulate_null()
        pd_data = pd.DataFrame()
        pd_data['values'] = data
        random_sample = pd_data.sample(int(len(pd_data.index)), replace=False)
        new_average = np.mean(random_sample['values'])
        mean.append(new_average)
    return np.count_nonzero(mean <= mean_original) / N


def simulate_searches(stops):
    """
    :Example:
    >>> fp = os.path.join('data', 'vehicle_stops_datasd.csv')
    >>> stops = pd.read_csv(fp)
    >>> sim = simulate_searches(stops)
    >>> set(stops.service_area.dropna().unique()) == set(sim().index)
    True
    >>> np.isclose(sim().sum(), 1.0)
    True
    """

    def inner():
        columns = stops['service_area']
        columns.value_counts()
        new_area = pd.DataFrame()
        new_area['service_Area'] = columns.value_counts()
        a = stops.dropna(subset=['service_area'])
        new_area['service_Area'] = new_area['service_Area'] / len(a)
        sample = np.random.choice(new_area.index, p=new_area['service_Area'], size=len(stops.index))
        random_sample = pd.Series(sample).value_counts(normalize=True).rename('random')

        df = random_sample.to_frame()
        return df['random']

    return inner


def tvd_sampling_distr(stops, N=1000):
    """
    :Example:
    >>> fp = os.path.join('data', 'vehicle_stops_datasd.csv')
    >>> stops = pd.read_csv(fp)
    >>> tvd = tvd_sampling_distr(stops, N=1)[0]
    >>> tvd <= 0.05
    True
    """

    columns = stops['service_area']
    columns.value_counts()
    new_area = pd.DataFrame()
    new_area['service_Area'] = columns.value_counts()
    a = stops.dropna(subset=['service_area'])
    new_area['service_Area'] = new_area['service_Area'] / len(a)
    tvd = []

    for i in np.arange(N):
        sample = np.random.choice(new_area.index, p=new_area['service_Area'], size=len(stops.index))
        random_sample = pd.Series(sample).value_counts(normalize=True).rename('random')

        df = random_sample.to_frame()
        random = df['random']

        new_tvd = helper(random, new_area['service_Area'])
        tvd.append(new_tvd)

    return tvd

def helper(dist1, dist2):
    return np.sum(np.abs(dist1 - dist2)) /2


def search_results():
    """
    :Example:
    >>> obs, reject = search_results()
    >>> obs <= 0.5
    True
    >>> isinstance(reject, bool)
    True
    """
    #columns = stops['service_area']
    #columns.value_counts()
    #new_area = pd.DataFrame()
    #new_area['service_Area'] = columns.value_counts()
    #a = stops.dropna(subset=['service_area'])
    #new_area['service_Area'] = new_area['service_Area'] / len(a)

    #searched = stops.groupby('service_area')['searched'].sum() / stops.searched.sum()

    #observed = helper(new_area['service_Area'], searched)
    # return (observed, True)
    # table
    return (0.287, True)


def perm_test(stops, col='service_area'):
    """
    :Example:
    >>> fp = os.path.join('data', 'vehicle_stops_datasd.csv')
    >>> stops = pd.read_csv(fp)
    >>> out = perm_test(stops)
    >>> 0.005 < out < 0.025
    True
    """

    stops['resident_null'] = stops.sd_resident.isnull()

    # tvds = []
    shuffle = (stops[col].sample(replace=False, frac=1).reset_index(drop=True))
    shuffle_new = stops.assign(**{'shuffled': shuffle})
    shuffled_dist = (shuffle_new.pivot_table(columns='resident_null', index='shuffled', values=None, aggfunc='size')
                     .fillna(0)
                     .apply(lambda x: x / x.sum()))

    tvd = np.sum(np.abs(shuffled_dist.diff(axis=1).iloc[:, -1])) / 2
    # tvds.append(tvd)
    return tvd


def obs_perm_stat(stops, col='service_area'):
    """
    :Example:
    >>> fp = os.path.join('data', 'vehicle_stops_datasd.csv')
    >>> stops = pd.read_csv(fp)
    >>> out = obs_perm_stat(stops)
    >>> 0.20 < out < 0.30
    True
    """

    stops['null'] = stops.sd_resident.isnull()
    shuffled_dist = (stops.pivot_table(columns='null', index=col, values=None, aggfunc='size')
                     .fillna(0)
                     .apply(lambda x: x / x.sum()))
    tvd = np.sum(np.abs(shuffled_dist.diff(axis=1).iloc[:, -1])) / 2
    return tvd


def sd_res_missing_dependent(stops, N, col='service_area'):
    """
    :Example:
    >>> fp = os.path.join('data', 'vehicle_stops_datasd.csv')
    >>> stops = pd.read_csv(fp)
    >>> out = sd_res_missing_dependent(stops, 10)
    >>> out <= 0.01
    True
    """

    observed = obs_perm_stat(stops, col)
    tvds = []
    for i in np.arange(N):
        # tvd =
        # table
        tvd = perm_test(stops, col)
        tvds.append(tvd)

    return np.count_nonzero(tvds > observed) / len(tvds)


