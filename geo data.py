
import pandas as pd
import numpy as np
import matplotlib.path as mpltPath
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
import json
import math
from scipy import stats


def best_transformation():
    """
    :Example:
    >>> best_transformation() in [1,2,3,4]
    True
    """

    new_list = []
    root_list = []
    for i in range(100):
        new_list.append(int(math.pow(i, 2)))

    for i in new_list:
        root_list.append(int(math.pow(i, 0.5)))

    df = pd.DataFrame(new_list, root_list, columns=['power_two'])
    df['root_power'] = root_list
    plt.subplot(2, 1, 1)
    plot_1 = df['power_two'].plot(kind='hist', color='blue')
    plt.subplot(2, 1, 2)
    plot_2 = df['root_power'].plot(kind='hist', color='orange')
    log_list = []
    for i in range(100):
        log_list.append(math.exp(i))
    df_log = pd.DataFrame(log_list, columns=['log_list'])
    df_log['log_transform'] = df_log['log_list'].apply(np.log)
    plt.subplot(2, 1, 1)
    plot_log1 = df_log['log_list'].plot(kind='hist', color='blue')
    plt.subplot(2, 1, 2)
    plot_log2 = df_log['log_transform'].plot(kind='hist', color='orange')


    passengers = pd.read_csv('data/passengers.csv')

    df_root_p = pd.DataFrame()
    df_root_p['passengers_root'] = passengers['Total_month'].apply(np.sqrt)
    df_root_p['year'] = passengers['Year_Month']

    df_log_p = pd.DataFrame()
    df_log_p['passengers_log'] = passengers['Total_month'].apply(np.log)
    df_log_p['year'] = passengers['Year_Month']

    df_log_p.plot(x='year', y='passengers_log', kind='hist', color='blue')

    df_root_p.plot(x='year', y='passengers_root', kind='hist', color='orange')

    return 2


def prediction(population):
    """

    :Example:
    >>> d = {}
    >>> d["Time_in_Days"] = [0, 40, 70, 110, 140, 171, 181]
    >>> d["Population"] = [10, 20, 60, 160, 325, 890, 1112]
    >>> df = pd.DataFrame(d)
    >>> pred, choice = prediction(df)
    >>> pred > 0
    True
    >>> choice in [1,2,3]
    True
    """

    population.plot(x='Time_in_Days', y='Population', kind='line')
    df_square = pd.DataFrame()
    df_square['time'] = population['Time_in_Days']
    df_square['square_root'] = population['Population'].apply(np.sqrt)


    df_log = pd.DataFrame()
    df_log['time'] = population['Time_in_Days']
    df_log['log'] = population['Population'].apply(np.log)
    slope_log, intercept_log, r_value_log, p_value_log, std_err_log = stats.linregress(df_log)


    prediction = slope_log * 400 + intercept_log

    un_log = math.exp(prediction)

    return (un_log, 3)



def latlong2geojson(route):
    """

    :Example:
    >>> path = os.path.join('data', 'aarons-day.txt')
    >>> data = pd.read_csv(path, sep='\\t')
    >>> gj = latlong2geojson(data)
    >>> types = set([x['geometry']['type'] for x in gj['features']])
    >>> types == set(['MultiLineString', 'Point'])
    True
    """
    lst = []
    for i in range(len(route['latitude'])):
        lst.append([route.latitude[i], route.longitude[i]])
    df = pd.DataFrame()
    df['long_lat'] = lst

    other_lst = []
    for i in range(len(df['long_lat'])):
        for j in range(2):
            other_lst.append([df.long_lat[j], df.long_lat[j + 1]])
            break
    a = df.sample(n=2)
    marker_lst = [a.iloc[0]['long_lat'], a.iloc[1]['long_lat']]

    # m = folium.Map(location = b)
    # point = folium.Marker(b).add_to(m)
    dictionary = ({'route': other_lst,
                   'marker': marker_lst})

    return dictionary


def trajectory_distance(filepath):
    """
    
    :Example:
    >>> path = os.path.join('data', '20081023025304.plt')
    >>> dist = trajectory_distance(path)
    >>> 0 <= dist
    True
    """
    path = pd.read_csv(filepath, skiprows=lambda x: x in range(6), header=None)
    df = pd.DataFrame()
    df['latitude'] = path[0]
    df['longitude'] = path[1]
    rads = df.applymap(np.deg2rad)

    lat1 = rads['latitude']
    long1 = rads['longitude']

    # To take successive difference, shift arrays up by one.
    lat2 = rads['latitude'].iloc[1:].reset_index(drop=True)
    long2 = rads['longitude'].iloc[1:].reset_index(drop=True)

    lat = lat2 - lat1
    long = long2 - long1

    inside = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(long * 0.5) ** 2
    a = 2 * 6371.0088 * np.arcsin(np.sqrt(inside)) * 0.621
    return a.sum()


def point_in_state(coord, state, geojson):
    """
    
    :Example:
    >>> path = os.path.join('data', 'states.geojson')
    >>> geojson = json.load(open(path))
    >>> coord = [32.881, -117.237]
    >>> point_in_state(coord, 'CA', geojson)
    True
    >>>
    """

    lst = []
    for feature in geojson['features']:
        if feature['id'] == state:
            surrounding = feature['geometry']['coordinates']
            # print(a)
            for i in surrounding:
                for j in i:
                    lst.append(j)
            lst_a = np.array(lst)
            # path = []
            if len(lst_a[0]) <= 2:
                lst_a[:, [0, 1]] = lst_a[:, [1, 0]]
                path = mpltPath.Path(lst_a)
                return path.contains_point(coord)

            else:
                for k in lst_a:
                    k = np.array(k)
                    k[:, [0, 1]] = k[:, [1, 0]]
                    path = mpltPath.Path(k)
                    if path.contains_point(coord):
                        return True

    return False


def label_state(coord, geojson):
    """

    :Example:
    >>> path = os.path.join('data', 'states.geojson')
    >>> geojson = json.load(open(path))
    >>> coord = [32.881, -117.237]
    >>> label_state(coord, geojson)
    'CA'
    """

    for i in geojson['features']:
        if point_in_state(coord, i['id'], geojson):
            return i['id']
    else:
        return None




