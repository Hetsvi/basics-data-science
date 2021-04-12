
import pandas as pd
import numpy as np
import glob
import os

def major_drop(df):
    """
    major_drop removes the columns specified in the write up
    """

    columns = [i for i in df.columns if 'Lab' in i]
    columns += [i for i in df.columns if 'Quiz' in i]
    columns += [i for i in df.columns if 'Project' in i]
    columns += [i for i in df.columns if 'Midterm' and 'Lateness' in i]
    columns += [i for i in df.columns if 'Final' and 'Lateness' in i]
    columns += [i for i in df.columns if 'Total Lateness' in i]

    columns = list(set(columns))
    df = df.drop(columns, axis=1)
    return df


def merged_exams(df):
    """
    major_drop removes the columns specified in the write up
    """

    df.replace(np.NaN, 0, inplace=True)
    df["Midterm Grades"] = df["Midterm V1"] + df["Midterm V2"] + df["Midterm V3"]
    df["Final Grades"] = df["DSC20 Final V2"] + df["DSC20 Final"]
    a = ["Midterm V1", "Midterm V2", "Midterm V3", "DSC20 Final", "DSC20 Final V2"]
    df = df.drop(a, axis=1)
    return df


def read_all(dirname):
    """
    Finds and reads .csv files from the timeData directory
    """
    file = os.listdir(dirname)
    lst = []
    for i in file:
        f = dirname + "/" + i
        df = pd.read_csv(f)
        lst.append(df)
    return lst


def extract_and_create(hws):
    """
    """
    #file = read_all(hws)
    for i in hws:
        i['Lateness (H:M:S)'].replace(np.NaN, '00:00:00', inplace=True)

    type_each = []
    master = []
    for i in hws:
        # print(i)
        for j in i['Lateness (H:M:S)']:
            if (j > "24:00:00" and j < '48:00:00'):
                type_each.append(2)
            elif (j > "00:00:00" and j < "24:00:00"):
                type_each.append(1)
            else:
                type_each.append(0)
        i["type"] = type_each
        master.append(type_each)
        type_each = []

    penalty_20 = []
    penalty_50 = []

    df = pd.DataFrame(master)
    for i in df.columns:
        a = df[i].tolist().count(1)
        b = df[i].tolist().count(2)

        penalty_20.append(a)
        penalty_50.append(b)

    new_df = {'Penalty_20': penalty_20,
              'Penalty_50': penalty_50}

    return pd.DataFrame(new_df)




def compute_stats(fh1, fh2, fh3):
    """
    >>> fh1, fh2, fh3 = open('linkedin1.csv'), open('linkedin2.csv'), open('linkedin3.csv')
    >>> out = compute_stats(fh1, fh2, fh3)
    >>> set(map(lambda x:isinstance(x, str), out)) == {True}
    True
    >>> len(out)  # first name, job, slogan, animal
    4
    """

    a = pd.read_csv(fh1)
    b = pd.read_csv(fh2)
    c = pd.read_csv(fh3)
    a = a.drop('Unnamed: 0', axis=1)
    b = b.rename(columns={'first_name': 'firstname'})
    b = b.rename(columns={'favorite_animal': 'favoriteanimal'})
    c = c.rename(columns={'COMPANY': 'company'})
    c = c.rename(columns={'OTHER': 'other'})
    c = c.rename(columns={'JOB': 'job'})
    c = c.rename(columns={'FIRSTNAME': 'firstname'})
    c = c.rename(columns={'SLOGAN': 'slogan'})
    c = c.rename(columns={'FAVORITEANIMAL': 'favoriteanimal'})

    combined = pd.merge(a, b, how='outer')
    new_combined = pd.merge(combined, c, how='outer')

    name = new_combined.groupby(['firstname']).size().idxmax()
    job = new_combined.groupby(['job']).size().idxmax()
    slogan = new_combined.groupby(['slogan']).size().idxmax()
    favorite = new_combined.groupby(['favoriteanimal']).size().idxmax()

    return [name, job, slogan, favorite]



def job_word_distribution(jobtitles):
    """
    >>> salaries = pd.read_csv('san-diego-2017.csv')
    >>> jobtitle = salaries['Job Title']
    >>> out = job_word_distribution(jobtitle)
    >>> 'Police' in out.index
    True
    >>> set(map(lambda x:x.count(' ') == 0, out.index)) == {True}
    True
    >>> (len(out) >= 500) and (len(out) <= 550) # number of distinct words
    True
    """

    jobtitles = jobtitles.str.cat(sep=' ')
    a = jobtitles.split()

    return pd.Series(a).value_counts()



def describe_salaries_by_job_type(salaries):
    """
    >>> salaries = pd.read_csv('san-diego-2017.csv')
    >>> out = describe_salaries_by_job_type(salaries)
    >>> (out.columns == ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']).all()
    True
    """
    job_types = ['Police', 'Fire', 'Libr', 'Rec', 'Grounds', 'Lifeguard', 'Water', 'Equip', 'Utility', 'Clerical',
                 'Administrative', 'Sanitation', 'Principal', 'Public', 'Dispatcher']
    lst = []
    for i in range(salaries['Job Title'].count()):
        # for j in salaries['Job Title'][i].split():
        for j in job_types:
            if j in salaries['Job Title'][i]:
                lst.append(j)
            else:
                lst.append('other')
            # if j in job_types:
            # a = salaries['Total Pay'].describe
    # salaries['new'] = lst
    salaries['new'] = pd.Series(lst)
    common = salaries.groupby('new')
    return common['Total Pay'].describe()



def std_salaries_by_job_type(salaries):
    """
    >>> salaries = pd.read_csv('san-diego-2017.csv')
    >>> out = std_salaries_by_job_type(salaries)
    >>> set(out.columns) == set(['Base Pay', 'Overtime Pay', 'Total Pay', 'Job Type'])
    True
    >>> np.all(abs(out.select_dtypes(include='number').mean()) < 10**-7)  # standard units should average to 0!
    True
    """

    job_types = ['Police', 'Fire', 'Libr', 'Rec', 'Grounds', 'Lifeguard', 'Water', 'Equip', 'Utility', 'Clerical',
                 'Administrative', 'Sanitation', 'Principal', 'Public', 'Dispatcher']
    lst = []
    for i in range(salaries['Job Title'].count()):
        # for j in salaries['Job Title'][i].split():
        for j in job_types:
            if j in salaries['Job Title'][i]:
                lst.append(j)
            else:
                lst.append('other')
            # if j in job_types:
            # a = salaries['Total Pay'].describe
    # salaries['new'] = lst
    salaries['new'] = pd.Series(lst)
    common = salaries.groupby('new')
    x = common.transform(lambda x: (x - x.mean()) / x.std())
    df = {'Job Type': salaries['new'],
          'Base Pay': x['Base Pay'],
          'Overtime Pay': x['Overtime Pay'],
          'Total Pay': x['Total Pay']}

    return pd.DataFrame(df)


def bucket_total_pay(totpay):
    """
    >>> salaries = pd.read_csv('san-diego-2017.csv')
    >>> out = bucket_total_pay(salaries['Total Pay'])
    >>> set(np.unique(out)) == set(range(1,11))
    True
    >>> np.all(np.abs(np.histogram(out)[0] - out.size/10) < 1)  # equal bin sizes!
    True
    """
    bins = (np.percentile(totpay, np.arange(10, 101, 10))).astype(int)
    lst = []
    for i in range(totpay.count()):
        if totpay[i] <= bins[0]:
            lst.append(1)
        if totpay[i] <= bins[1] and totpay[i] > bins[0]:
            lst.append(2)
        if totpay[i] <= bins[2] and totpay[i] > bins[1]:
            lst.append(3)
        if totpay[i] <= bins[3] and totpay[i] > bins[2]:
            lst.append(4)
        if totpay[i] <= bins[4] and totpay[i] > bins[3]:
            lst.append(5)
        if totpay[i] <= bins[5] and totpay[i] > bins[4]:
            lst.append(6)
        if totpay[i] <= bins[6] and totpay[i] > bins[5]:
            lst.append(7)
        if totpay[i] <= bins[7] and totpay[i] > bins[6]:
            lst.append(8)
        if totpay[i] <= bins[8] and totpay[i] > bins[7]:
            lst.append(9)
        if totpay[i] <= bins[9] and totpay[i] > bins[8]:
            lst.append(10)
        # if totpay[i] >= bins[9]:
        #   lst.append(10)

    return pd.Series(lst)

def mean_salary_per_decile(salaries):
    """
    >>> salaries = pd.read_csv('san-diego-2017.csv')
    >>> out = mean_salary_per_decile(salaries)
    >>> len(out) == 10
    True
    >>> 50000 <= out[5] <= 60000
    True
    """

    totpay = salaries['Total Pay']
    bins = (np.percentile(totpay, np.arange(10, 101, 10))).astype(int)
    lst = []
    lst2 = []
    lst3 = []
    lst4 = []
    lst5 = []
    lst6 = []
    lst7 = []
    lst8 = []
    lst9 = []
    lst10 = []
    for i in range(totpay.count()):
        if totpay[i] <= bins[0]:
            lst.append(totpay[i])
        if totpay[i] <= bins[1] and totpay[i] > bins[0]:
            lst2.append(totpay[i])
        if totpay[i] <= bins[2] and totpay[i] > bins[1]:
            lst3.append(totpay[i])
        if totpay[i] <= bins[3] and totpay[i] > bins[2]:
            lst4.append(totpay[i])
        if totpay[i] <= bins[4] and totpay[i] > bins[3]:
            lst5.append(totpay[i])
        if totpay[i] <= bins[5] and totpay[i] > bins[4]:
            lst6.append(totpay[i])
        if totpay[i] <= bins[6] and totpay[i] > bins[5]:
            lst7.append(totpay[i])
        if totpay[i] <= bins[7] and totpay[i] > bins[6]:
            lst8.append(totpay[i])
        if totpay[i] <= bins[8] and totpay[i] > bins[7]:
            lst9.append(totpay[i])
        if totpay[i] <= bins[9] and totpay[i] > bins[8]:
            lst10.append(totpay[i])

    return pd.Series([np.mean(lst), np.mean(lst2), np.mean(lst3), np.mean(lst4), np.mean(lst5), np.mean(lst6),
                      np.mean(lst7), np.mean(lst8), np.mean(lst9), np.mean(lst10)],
                     index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])



def robo_table(phones):
    """
    >>> phones = pd.read_csv('phones.csv')
    >>> out = robo_table(phones)
    >>> set(out.columns) == set(['id', 'first_name', 'last_name', 'phone'])
    True
    >>> _ = out.phone.dropna().astype(int)
    """
    df = phones

    phone = df['cell_phone'].fillna(df['home_phone'])
    phone = phone.fillna(df['work_phone'])
    phone = phone.apply(str)

    a = phone.notnull()
    for i in range(phone.count()):
        if a[i]:
            filter(lambda x: x.isdigit(), phone[i])
            phone[i] = str(phone[i])
            phone[i] = phone[i].replace('.0', '')

    newdf = {'id': df.id,
             'first_name': df.first_name,
             'last_name': df.last_name,
             'phone': phone}

    return pd.DataFrame(newdf)


import re
def read_names(dirname):
    """
    >>> out = read_names('names')
    >>> set(out.columns) == set(['first_name', 'sex', 'number', 'year'])
    True
    >>> out.year.nunique()
    138
    """

    file = os.listdir(dirname)
    # print(file)
    lst = []
    for i in file:
        f = dirname + "/" + i
        columns = ['first_name', 'sex', 'number']
        df = pd.read_csv(f, names=columns, header=None)
        year = re.search('yob(.*).txt', i).group(1)
        # year = re.search('yob(.*).txt', i)
        df['year'] = year
        lst.append(df)

    frame = pd.concat(lst, axis=0, ignore_index=True)

    return frame



