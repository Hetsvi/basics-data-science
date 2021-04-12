
import os
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium


BASIC_COLS = ['UNITID', 'OPEID', 'OPEID6', 
              'INSTNM', 'ZIP', 'LATITUDE', 
              'LONGITUDE', 'CONTROL', 'PREDDEG', 'UGDS']


def translation_dict(datadict):
    """

    translation_dict  outputs a dictionary satisfying 
    the following conditions:
    - The keys are the column names of colleges that are 
    strings encoded as integers (i.e. columns for which 
    VALUE and LABEL in datadict are non-empty).
    - The values are also dictionaries; each has keys 
    given by VALUE and values LABEL.

    """
    newdict = datadict.dropna(subset=['VALUE', 'LABEL'])
    newdict = newdict.fillna(method = 'ffill')
    labels = newdict['VARIABLE NAME'].unique()

    newdict = newdict.set_index('VARIABLE NAME')
    newdict = (newdict.groupby(level = 'VARIABLE NAME').apply
                   (lambda x: dict(zip(x['VALUE'], x['LABEL']))).to_dict())
    return newdict


def basic_stats(college):
    """

    basic_stats takes in college and returns 
    a Series of the above statistics index by:
    ['num_schools', 'num_satellite', 'num_students', 'avg_univ_size']


    """
    df = college.copy()
    num_schools = len(college.OPEID.unique().tolist())
    df["dup"] = college.OPEID6.duplicated(False)
    df = df.loc[df['dup'] == True]
    df = df.groupby(['OPEID6']).size()
    num_satellite = len(df)
    num_students = college.UGDS.sum()
    avg_univ_size = college.UGDS.mean()
    df = pd.DataFrame(index = ['num_schools', 'num_satellite', 'num_students', 'avg_univ_size'])
    a = [num_schools, num_satellite, num_students, avg_univ_size]
    df['columns'] = a
    new = []
    for i in df['columns']:
        i = int(i)
        new.append(i) 
    
    df['new'] = new
    return df['new']


def plot_school_sizes(college):
    """

    :Example:
    >>> college_path = os.path.join('data', 'MERGED2016_17_PP.csv')
    >>> college = pd.read_csv(college_path)
    >>> ax = plot_school_sizes(college)
    >>> ax.get_ylabel()
    'Frequency'
    """
    num = college['UGDS']
    fig, ax = plt.subplots(1, 1)
    ax.set_ylabel('Frequency')
    ax.set_yscale('log')
    ax.hist(num, bins = 20)
    return ax



def num_of_small_schools(college, k):
    """

    :Example:
    >>> college_path = os.path.join('data', 'MERGED2016_17_PP.csv')
    >>> college = pd.read_csv(college_path)
    >>> nschools = len(college)
    >>> out = num_of_small_schools(college, nschools - 1)
    >>> out == (len(college) - 1)
    True
    >>> import numbers
    >>> isinstance(num_of_small_schools(college, 2), numbers.Integral)
    True
    """
    college_ascending = college.UGDS.sort_values(ascending=False).reset_index()
    sum_k =0
    for i in range(k):
        sum_k = sum_k + college_ascending['UGDS'][i]
    
    small_sum = 0
    college_asc_opp = college.UGDS.sort_values().reset_index()
    index = 0
    counter = -1
    
    while(small_sum <= sum_k):
        small_sum = small_sum + college_asc_opp['UGDS'][index]
        index = index + 1
        counter = counter + 1 
        
    return counter



def col_pop_stats(college, col):
    """

    :Example:
    >>> college_path = os.path.join('data', 'MERGED2016_17_PP.csv')
    >>> college = pd.read_csv(college_path)
    >>> out = col_pop_stats(college, "PREDDEG")
    >>> (out.columns == ['size', 'sum', 'mean', 'median']).all()
    True
    >>> (out.index == [1,2,3]).all()
    True
    >>> (out > 0).all().all()
    True
    """
    a = college[col].unique().sort()
    df = pd.DataFrame(index= a)
    size = college.groupby(col)['OPEID'].count()
    df['size'] = size
    sumc = college.groupby(col)['UGDS'].sum()
    df['sum'] = sumc
    mean = college.groupby(col)['UGDS'].mean()
    df['mean'] = mean
    median = college.groupby(col)['UGDS'].median()
    df['median'] = median
    
    return df


def col_pop_stats_plots(stats, datadict):
    """

    :Example:
    >>> datadict_path = os.path.join('data', 'CollegeScorecardDataDictionary.xlsx')
    >>> datadict = pd.read_excel(datadict_path, sheet_name='data_dictionary')
    >>> college_path = os.path.join('data', 'MERGED2016_17_PP.csv')
    >>> college = pd.read_csv(college_path)
    >>> out = col_pop_stats(college, "PREDDEG")
    >>> ax = col_pop_stats_plots(out, datadict)
    >>> len(ax)
    4
    >>> ax[-1].get_title()
    'median'
    """
    newdict = translation_dict(datadict)
    new = stats.index.to_series().map(newdict['PREDDEG'])
    stats = stats.set_index(new)
    num_row, num_col = stats.shape
    fig, axarr = plt.subplots(num_col, sharex = True)
    for i in range(num_col):
        lst = newdict['PREDDEG'][i]
        axarr[i].bar(stats.index, stats.iloc[:,i])
        axarr[i].set_title(stats.columns.values.tolist()[i])
    plt.xlabel('PREDEG')
    plt.show()
    return axarr


def control_preddeg_stats(college, f):
    """

    :Example:
    >>> college_path = os.path.join('data', 'MERGED2016_17_PP.csv')
    >>> college = pd.read_csv(college_path)
    >>> out = control_preddeg_stats(college, lambda x:1)
    >>> (out == 1).all().all()
    True
    >>> out.index.name
    'CONTROL'
    >>> out.columns.name
    'PREDDEG'
    """
    sumc = college.groupby(['CONTROL', 'PREDDEG'])['UGDS'].apply(f)
    df = pd.DataFrame(sumc)
    df = df.reset_index()
    new = pd.DataFrame(index = df.PREDDEG)
    new['CONTROL'] = df.CONTROL.tolist()
    new['UGDS'] = df.UGDS.tolist()
    new = new.pivot_table(index = 'CONTROL', columns= new.index, values = 'UGDS')
    return new


def control_preddeg_stats_plot(out, datadict):
    """
    :Example:
    >>> datadict_path = os.path.join('data', 'CollegeScorecardDataDictionary.xlsx')
    >>> datadict = pd.read_excel(datadict_path, sheet_name='data_dictionary')
    >>> college_path = os.path.join('data', 'MERGED2016_17_PP.csv')
    >>> college = pd.read_csv(college_path)
    >>> out = control_preddeg_stats(college, lambda x:1)
    >>> ax = control_preddeg_stats_plot(out, datadict)
    >>> ax.get_children()[0].get_height()
    1
    >>> ax.get_xlabel()
    'CONTROL'
    """
    newdict = translation_dict(datadict)
    new = out.index.to_series().map(newdict[out.index.name])
    out = out.set_index(new)
    ax = out.plot.bar()

    return ax
    



def scatterplot_us(college):
    """

    :Example:
    >>> college_path = os.path.join('data', 'MERGED2016_17_PP.csv')
    >>> college = pd.read_csv(college_path)
    >>> ax = scatterplot_us(college)
    >>> ax.get_xlabel()
    'LONGITUDE'
    >>> ax.get_title()
    'Undergraduate Institutions'
    >>>
    

    """
    states = ['AL', 'AZ', 'AR','CA','CO','CT','DE','FL','GA','ID','IL','IN','IA',
              'KS','KY','LA','ME','MD','MA','MI','MN','MS','MO', 'MT','NE','NV','NH',
              'NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN',
              'TX','UT','VT','VA','WA','WV','WI','WY', 'DC']
    lower = college[college['STABBR'].isin(states)]
    lower.CONTROL = lower['CONTROL'].apply(lambda x: str(x)+ '_')
    ax = sns.scatterplot(x = 'LONGITUDE', y = 'LATITUDE', data = lower, hue = 'CONTROL'
                        ,size = 'UGDS')
    ax.set_title('Undergraduate Institutions')
    plt.xlim(-135, -60)
    plt.ylim(15, 55)
    return ax


def plot_folium():
    """

    :Example:
    >>> d = plot_folium()
    >>> isinstance(d, dict)
    True
    >>> 'geo_data' in d.keys()
    True
    >>> isinstance(d.get('key_on'), str)
    True
    """
    college_path = os.path.join('data', 'MERGED2016_17_PP.csv')
    college = pd.read_csv(college_path)
    states = pd.read_csv('data/population-2017.csv')
    state_geo = os.path.join('data/gz_2010_us_040_00_5m.json')
    number_college = college.groupby(['STABBR'])['UGDS'].sum()
    number_college = pd.DataFrame(number_college)
    number_residents = states.rename(columns = {"STATE" : "STABBR"}).set_index('STABBR')
    joined = number_college.join(number_residents)
    joined['ratio'] = joined.UGDS / joined.POP
    joined = joined.reset_index()
    names = {"AL":"Alabama", "AK":"Alaska", "AZ":"Arizona","AR":"Arkansas",
             "CA":"California", "CO":"Colarado", "CT":"Connecticut",
             "DE":"Delaware", "DC":"District Of Columbia", "FL":"Florida",
             "GA":"Georgia", "HI":"Hawaii", "ID":"Idaho", "IL":"Illinois",
             "IN":"Indiana","IA":"Iowa", "KS":"Kansas",  "KY":"Kentucky",
             "LA":"Louisiana","ME":"Maine", "MD":"Maryland", "MA":"Massachusetts",
             "MI":"Michigan", "MN":"Minnesota", "MS":"Mississippi", "MO":"Missouri",
             "MT":"Montana", "NE":"Nebraska", "NV":"Nevada", "NH":"New Hampshire",
             "NJ":"New Jersey", "NM":"New Mexico", "NY":"New York", "NC":"North Carolina",
             "ND":"North Dakota", "OH":"Ohio", "OK":"Oklahoma", "OR":"Oregon",
             "PA":"Pennsylvania","RI":"Rhode Island", "SC":"South Carolina",
             "SD":"South Dakota", "TN":"Tennessee", "TX":"Texas", "UT":"Utah",
             "VT":"Vermont", "VA":"Virginia", "WA":"Washington", "WV":"West Virginia",
             "WI":"Wisconsin", "WY":"Wyoming"}
    joined['states'] = joined['STABBR'].map(names)


    m = folium.Map(location=[48, -102], zoom_start = 3)
    folium.GeoJson(state_geo, name = "states").add_to(m)
    
    m.choropleth(
    geo_data= state_geo,
    name='choropleth',
    data= joined,
    columns=['states','ratio'],
    key_on='properties.NAME',
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='College per state (%)'
    )
    
    m.save('pct_students_by_state.html')
    mdict = {"geo_data" : state_geo, 
            "name" : 'choropleth',
            "data" : joined,
            "columns" : ['STABBR','ratio'],
            "key_on" : 'properties.STATE',
            "fill_color" : 'BuPu',
            "fill_opacity" : 0.7,
            "line_opacity" : 0.2,
            "legend_name" : 'College per state (%)'}

    return mdict



def control_type_by_state(college):
    """

    :Example:
    >>> college_path = os.path.join('data', 'MERGED2016_17_PP.csv')
    >>> college = pd.read_csv(college_path)
    >>> out = control_type_by_state(college)
    >>> len(out)
    49
    >>> np.allclose(out.sum(axis=1), 1)
    True
    """
    lower = ['AL', 'AZ', 'AR','CA','CO','CT','DE','FL','GA','ID','IL','IN','IA',
              'KS','KY','LA','ME','MD','MA','MI','MN','MS','MO', 'MT','NE','NV','NH',
              'NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN',
              'TX','UT','VT','VA','WA','WV','WI','WY', 'DC']
    states = college[college['STABBR'].isin(lower)]
    states = states.groupby(['STABBR', 'CONTROL']).size().reset_index()
    states = states.groupby(['STABBR', 'CONTROL']).agg({0:'sum'})
    percent = states.groupby(level=0).apply(lambda x: x/float(x.sum())).reset_index()
    states = percent.pivot(index = 'STABBR', columns = 'CONTROL', values = 0)
    return states


def tvd_states(college):
    """
    
    :Example:
    >>> college_path = os.path.join('data', 'MERGED2016_17_PP.csv')
    >>> college = pd.read_csv(college_path)
    >>> out = tvd_states(college)
    >>> len(out)
    49
    >>> 'NV' in out.index[:5]
    True
    >>> 'OR' in out.index[-5:]
    True
    """
    lower = ['AL', 'AZ', 'AR','CA','CO','CT','DE','FL','GA','ID','IL','IN','IA',
              'KS','KY','LA','ME','MD','MA','MI','MN','MS','MO', 'MT','NE','NV','NH',
              'NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN',
              'TX','UT','VT','VA','WA','WV','WI','WY', 'DC']
    states = college[college['STABBR'].isin(lower)]
    states = states.groupby(['STABBR', 'CONTROL']).size().reset_index()
    states = states.drop(['STABBR'], axis = 1)
    states = states.groupby(['CONTROL']).sum().reset_index() 
    states = states.set_index('CONTROL')
    total = states[0].sum()
    a = states[0]/ total 
    a =  pd.DataFrame(a)
    a = a.T
    a = pd.concat([a]*49)
    a['STATE'] = lower 
    a = a.set_index('STATE')
    total_college = control_type_by_state(college)
    diff = total_college.subtract(a)
    diff_abs = np.abs(diff)
    tvd = np.sum(diff_abs, axis = 1) / 2
    tvd = tvd.sort_values(ascending=False)    
    return tvd



def num_subjects(college):
    """

    :Example:
    >>> college_path = os.path.join('data', 'MERGED2016_17_PP.csv')
    >>> college = pd.read_csv(college_path)
    >>> out = num_subjects(college)
    >>> len(out) == len(college)
    True
    >>> out.nunique()
    34
    """
    cols = [x for x in college.columns if 'PCIP' in x]
    df = college[cols]
    subjects = df.astype(bool).sum(axis=1)
    return subjects


def subject_counts(college):
    """

    :Example:
    >>> college_path = os.path.join('data', 'MERGED2016_17_PP.csv')
    >>> college = pd.read_csv(college_path)
    >>> out = subject_counts(college)
    >>> len(out)
    34
    >>> out.loc[0].sum() == 3060
    True
    """
    data = college.copy()
    data['num_subjects'] = num_subjects(college)
    table = pd.pivot_table(data, index = 'num_subjects', columns = 'CONTROL', values = 'UGDS',
                          aggfunc = np.sum)
    return table


def create_specialty(college, datadict):
    """

    :Example:
    >>> datadict_path = os.path.join('data', 'CollegeScorecardDataDictionary.xlsx')
    >>> datadict = pd.read_excel(datadict_path, sheet_name='data_dictionary')
    >>> college_path = os.path.join('data', 'MERGED2016_17_PP.csv')
    >>> college = pd.read_csv(college_path)
    >>> out = create_specialty(college, datadict)
    >>> len(out.columns) == len(college.columns) + 1
    True
    >>> 'Psychology' in out['SPECIALTY'].unique()
    True
    """
    data = college.copy()
    cols = [x for x in college.columns if 'PCIP' in x]
    df = college[cols]
    cols = df.columns
    subjects = df.apply(lambda x: x > 0)
    subjects = subjects.apply(lambda x: list(cols[x.values]), axis = 1)
    specialty = [x[0] if len(x) == 1 else np.nan for x in subjects]
    data['SPECIALTY'] = specialty
    translate = datadict.dropna(subset = ['LABEL', 'VARIABLE NAME'])
    translate = translate[translate['VARIABLE NAME'].str.contains('PCIP')]
    translate = dict(zip(translate['VARIABLE NAME'], translate['LABEL']))
    data = data.replace({'SPECIALTY':translate})
    return data


