import urllib
import matplotlib.dates as mdates
#matplotlib.dates as mdates, which is useful for converting date stamps to dates that matplotlib can understand.

def graph_data(stock):
    # Unfortunately, Yahoo's API is no longer available
    # feel free to adapt the code to another source, or use this drop-in replacement.
    stock_price_url = 'https://pythonprogramming.net/yahoo_finance_replacement'

    source_code = urllib.request.urlopen(stock_price_url).read().decode()

    stock_data = []
    split_source = source_code.split('\n')

    for line in split_source[1:]:
        split_line = line.split(',')
        if len(split_line) == 7:   #To do this, we'll just use some rudimentary filtration, checking to make sure there are 6 data points per line, and then making sure the term "values" isn't in the line.
            if 'values' not in line:
                stock_data.append(line)
                date, closep, highp, lowp, openp, adj_closep, volume = np.loadtxt(stock_data,
                                                                                  delimiter=',',
                                                                                  unpack=True,
                                                                                  # %Y = full year. 2015
                                                                                  # %y = partial year 15
                                                                                  # %m = number month
                                                                                  # %d = number day
                                                                                  # %H = hours
                                                                                  # %M = minutes
                                                                                  # %S = seconds
                                                                                  # 12-06-2014
                                                                                  # %m-%d-%Y
                                                                                  converters={
                                                                                      0: bytespdate2num('%Y-%m-%d')})
