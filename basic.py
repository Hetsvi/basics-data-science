
import pandas as pd
import numpy as np

def consecutive_ints(ints):
    """
    consecutive_ints tests whether a list contains two 
    adjacent elements that are consecutive integers.
    """

    if len(ints) == 0:
        return False

    for k in range(len(ints) - 1):
        diff = abs(ints[k] - ints[k+1])
        if diff == 1:
            return True

    return False


def median(nums):
    """
    median takes a non-empty list of numbers, 
    returning the median element of the list. 
    If the list has even length, it should return
    the mean of the two elements in the middle.
    """

    if len(nums) == 0:
        return 0

    sort = sorted(nums)

    if(len(nums) % 2 ):
        return sort[(len(nums) - 1) // 2]

    else:
        return (sort[(len(nums) - 1) // 2] + sort[((len(nums) - 1) // 2) +1])/2

    return 0


def common(a,b):

    while b != 0:
        if a == b:
            return a
        if a > b:
            return common(a-b, b)
        else:
            return common(a, b-a)

    return a


def lcm3(a, b, c):
    """
    lcm3 returns the least common multiple of a,b,c.
    """

    lcm2 = (a*b) // common(a, b)

    lcm = (lcm2 * c) // common(lcm2, c)

    return lcm


def same_diff_ints(ints):
    """
    same_diff_ints tests whether a list contains
    two list elements i places apart, whose distance 
    as integers is also i.
    """
    for first in range (0, len(ints), 1):
        for second in range (0, len(ints), 1):
            math_difference = abs(ints[first] - ints[second])
            index_difference = second - first
            if index_difference != 0 and math_difference == index_difference:
                return True

    return False


def prefixes(s):
    """
    prefixes returns a string of every 
    consecutive prefix of the input string.
    """
    #lists = [s[i] for i in range(0, len(s), 1)]

    word = ''
    skip = 0
    while skip <= len(s)-1:
        for i in range(0, skip, 1):
            word = word + (s[i])

        skip = skip + 1

    word = word + s

    return word


def evens_reversed(N):
    """
    evens_reversed returns a string containing 
    all even integers from  1  to  N  (inclusive)
    in reversed order, separated by spaces. 
    Each integer is zero padded.

    """
    reverse = ""
    first = N
    for i in range (N, 0, -1):
        if first > 9:
            if i % 2 == 0:
                reverse = reverse + '%02d' % i + " "

        else:
            if i % 2 == 0:
                reverse = reverse + str(i) + " "

    #if first > 9:
        #reverse = reverse[0: -1]
    #else:
        #reverse = reverse.replace("", " ")[1: -1]
    reverse = reverse[0: -1]
    return reverse

def last_chars(fh):
    """
    last_chars takes a file object and returns a 
    string consisting of the last character of the line.
    """
    string = ""

    for line in fh:
        string = string + line[-2:]
        string = string.strip('\n')
    return string



def cnt_values(s):
    """
    cnt_values returns counts of all two 
    character combinations in string s.
    """
    s = s.lower()
    count = dict()

    for i in range(0, len(s)-1, 1):
        if s[i].isalpha() == True and s[i+1].isalpha() == True:
            count[s[i] + s[i+1]] = count.get(s[i] + s[i+1], 0) + 1

    return count


def list_cnts(d):
    """
    list_cnts takes in a dictionary, as described above,
    and returns a list of the top 5 most common 
    letter combinations in descending order.
    """
    sort_list = sorted(d.items(), key= d.get(1), reverse=False)
    list_val = []
    for i in range(0, 4 and len(sort_list), 1):
        list_val.append(sort_list[i][0])

    return list_val


def airport_arrival_stats(aircode):
    """
    airport_arrival_stats that takes in an airport 
    code `aircode` and outputs a list with the 
    following quantities, in the following order:
    - number of arriving flights to `aircode`.
    - average flight delay of arriving flights to `aircode`.
    - median flight delay of arriving flights to `aircode`.
    - the airline code of the airline that most often 
    arrives to `aircode`.
    - the proportion of arriving flights to `aircode` that 
    are cancelled in July 2015.
    - the airline code of the airline with the longest flight 
    delay among all flights arriving to `aircode`.
    """
    airport_arrival = pd.read_csv('flights.csv')
    flight = airport_arrival.loc[airport_arrival['DESTINATION_AIRPORT'] == aircode]
    number_flights = len(flight)
    avg_delay = flight['DEPARTURE_DELAY'].mean()
    median_delay = flight['DEPARTURE_DELAY'].median()
    most_often = flight['AIRLINE'].mode()
    total_july= airport_arrival['DATE', 'CANCELLED'].loc[airport_arrival['DATE'].str.contains('2015-07')]['DATE'].count()
    cancel_july = total_july.loc[total_july['CANCELLED'] == 1]
    proportion = cancel_july/total_july
    long_delay = max(flight.loc[flight['DEPARTURE_DELAY']])
    long_delay = long_delay['AIRLINE'].to_string
    return[number_flights, avg_delay, median_delay, most_often, proportion, long_delay]


def cancel_cnt_airport(fh):
    """
    returns the number of cancelled flights for 
    each airport using out-of-memory techniques 
    (chunking; chunk-size=1000).
    """

    chunks = pd.read_csv(fh, chunksize=1000)
    next(chunks)

    list_air = pd.Series([])

    for chunk in chunks:
        filterd = filter('CANCELLED' == 1,chunk)
        one = filterd.groupby('ORIGIN_AIRPORT').sum()
        list_air.append(one)

    return list_air
