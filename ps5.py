# -*- coding: utf-8 -*-
# Problem Set 5: Modeling Temperature Change
# Name: Katie Kostecki
# Collaborators: None
# Time: 4:30

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import re

# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAINING_INTERVAL = range(1961, 2000)
TESTING_INTERVAL = range(2000, 2017)

##########################
#    Begin helper code   #
##########################

def standard_error_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.

    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d numpy array of values estimated by a linear
            regression model
        model: a numpy array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y)**2).sum()
    var_x = ((x - x.mean())**2).sum()
    SE = np.sqrt(EE/(len(x)-2)/var_x)
    return SE/model[0]


class Dataset(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Dataset instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature

        f.close()

    def get_daily_temps(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d numpy array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return np.array(temperatures)

    def get_temp_on_date(self, city, month, day, year):
        """
        Get the temperature for the given city at the specified date.

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified date and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year {} is not available".format(year)
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]

##########################
#    End helper code     #
##########################

    def get_yearly_averages(self, cities, years):
        """
        For each year in the given range of years, computes the average of the
        annual temperatures in the given cities.

        Args:
            cities: a list of the names of cities to include in the average
                annual temperature calculation
            years: a list of years to evaluate the average annual temperatures at

        Returns:
            a 1-d numpy array of floats with length = len(years). Each element in
            this array corresponds to the average annual temperature over the given
            cities for a given year.
        """

        # NOTE: TO BE IMPLEMENTED IN PART 4B OF THE PSET
        annual_temps = []
        
        #add up annual temps for each year
        for elem in range(len(years)):
        
            #generate list of averages of the average yearly temps for each city for every year
            avg_temps = []
            for city in range(len(cities)): 
                temps = self.get_daily_temps(cities[city], years[elem]) 
                avg_city_temps = np.array(temps).mean()
                avg_temps.append(avg_city_temps)
            annual_temps.append(np.array(avg_temps).mean())
        
        return np.array(annual_temps)
        
            
        
def linear_regression(x, y):
    """
    Calculates a linear regression model for the set of data points.

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points

    Returns:
        (m, b): A tuple containing the slope and y-intercept of the regression line,
                both of which are floats.
    """
    
    #calculate average of x and y
    x_avg = sum(x)/len(x)
    y_avg = sum(y)/len(y)
    
    #find the slope
    num = 0
    denom = 0
    for elem in range(len(x)): #sum up numerators and denomonators 
        x_diff = x[elem] - x_avg
        y_diff = y[elem] - y_avg
        num += (x_diff*y_diff) 
        denom += x_diff**2
  
    #compute slope and use mx + b formula to find y-intercept
    m = num/denom
    b = y_avg - (m*x_avg)
    
    return(m, b)
        
def squared_error(x, y, m, b):
    '''
    Calculates the squared error of the linear regression model given the set
    of data points.

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        m: The slope of the regression line
        b: The y-intercept of the regression line


    Returns:
        a float for the total squared error of the regression evaluated on the
        data set
    '''
    total_se = 0
    
    #calculates square error for each y value, adds them all up
    for elem in range(len(x)):
        y_val = y[elem]
        y_prime = b + (m*x[elem])
        total_se += (y_val - y_prime)**2 #square error is difference between y and y from the regression squared

    return total_se 


def generate_models(x, y, degrees):
    """
    Generates a list of polynomial regression models with degrees specified by
    degrees for the given set of data points

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        degrees: a list of integers that correspond to the degree of each polynomial
            model that will be fit to the data

    Returns:
        a list of numpy arrays, where each array is a 1-d numpy array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    models = []
    
    #get a different model for each degree
    for degree in degrees:
        fit = np.polyfit(x, y, degree) 
        models.append(fit)
        
    return models


def evaluate_models(x, y, models, display_graphs=False):
    """
    For each regression model, compute the R-squared value for this model and
    if display_graphs is True, plot the data along with the best fit curve.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (i.e. the model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        Degree of your regression model,
        R-squared of your model evaluated on the given data points,
        and standard error/slope (if this model is linear).

    R-squared and standard error/slope should be rounded to 4 decimal places.

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a numpy array storing the coefficients of
            a polynomial
        display_graphs: A boolean whose value specifies if the graphs should be
            displayed

    Returns:
        A list holding the R-squared value for each model
    """
    total_r_squared = []
    
    #get R^2 for each model
    for model in range(len(models)):
        pred_y = np.polyval(models[model], x)
        r_squared = r2_score(y, pred_y) #compute each R^2
        total_r_squared.append(r_squared)
        
        #make a graph for each model (when specified)
        if display_graphs == True:
            plt.plot(x,y, "bo") #blue dots for individual data
            plt.plot(x, pred_y, "r-") #red line for model
            plt.xlabel("Years")
            plt.ylabel("Temperature (Celcius)")
            
            if len(models[model]) == 2: #if linear, add SE/slope to title
                se_m = standard_error_over_slope(np.array(x), y, pred_y, models[model])
                plt.title("Fit of degree " + str(len(models[model])-1) + ", R2 = " + str(round(r_squared, 4)) + "\n SE/m = " + str(round(se_m, 4)))
            else:
                plt.title("Fit of degree " + str(len(models[model])-1) + ", R2 = " + str(round(r_squared, 4)))
            plt.show()
            
            
    return total_r_squared      


def find_extreme_trend(x, y, length, positive_slope):
    """
    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        length: the length of the interval
        positive_slope: a boolean whose value specifies whether to look for
            an interval with the most extreme positive slope (True) or the most
            extreme negative slope (False)

    Returns:
        a tuple of the form (i, j, m) such that the application of linear (deg=1)
        regression to the data in x[i:j], y[i:j] produces the most extreme
        slope m, with the sign specified by positive_slope and j-i = length.

        In the case of a tie, it returns the first interval. For example,
        if the intervals (2,5) and (8,11) both have slope 3.1, (2,5,3.1) should be returned.

        If no intervals matching the length and sign specified by positive_slope
        exist in the dataset then return None
    """
    # generate dictionary of intervals and corresponding slope
    pos_slopes = {}
    neg_slopes = {}
    for index in range(len(x) - length+1): #iterate through possible start years
        reg = linear_regression(x[index:index+length], y[index:index+length]) #get regression of specified length
        
        #add slopes that are positive to a dictionary. Key = years
        if reg[0] > 0:
            pos_slopes[(index, index+length)] = reg[0]
            
        #add up negative slopes to a different dictionary. Key = years
        elif reg[0] < 0:
            neg_slopes[(index, index+length)] = reg[0]
            
        #add slopes of 0 to both dictionaries
        else:
            pos_slopes[(index, index+length)] = reg[0]
            neg_slopes[(index, index+length)] = reg[0]
                                        
            
    #if finding positive slope, find max of positive dictionary
    if positive_slope:
        max_positive = None #returns None if no slopes are positive
        
        for interval in pos_slopes: 
            if max_positive is None: #first slope looked at will always be added
                max_positive = (interval[0], interval[1], pos_slopes[interval])
                
            elif abs(pos_slopes[interval] - max_positive[2]) <= 1e-8: #if they are the same, keep the first
                max_positive = max_positive
                
            elif pos_slopes[interval] - max_positive[2] > 1e-8: #if the new slope is bigger, replace is as the max
                max_positive = (interval[0], interval[1], pos_slopes[interval])
        return max_positive
    
    
    # if finding negative slope, find min of negative dictionary
    else:
        negative = None #return None if no negative slopes
        
        for interval in neg_slopes:
            if negative is None: #first slope looked at will always be added
                negative = (interval[0], interval[1], neg_slopes[interval])
                
            elif abs(neg_slopes[interval] - negative[2]) <= 1e-8: #if next slope is the same, keep the one already in
                negative = negative
                
            elif abs(neg_slopes[interval]) - abs(negative[2]) > (1e-8): #if new slope is smaller, replace it as most negative
                negative = (interval[0], interval[1], neg_slopes[interval])
        return negative

def find_all_extreme_trends(x, y):
    """
    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        
    Returns:
        a list of tuples of the form (i,j,m) such that the application of linear
        regression to the data in x[i:j], y[i:j] produces the most extreme
        positive OR negative slope m, and j-i=length. 

        The returned list should have len(x) - 1 tuples, with each tuple representing the
        most extreme slope and associated interval for all interval lengths 2 through len(x).
        If there is no positive or negative slope in a given interval length L (m=0 for all
        intervals of length L), the tuple should be of the form (0,L,None).

        The returned list should be ordered by increasing interval length. For example, the first 
        tuple should be for interval length 2, the second should be for interval length 3, and so on.

        If len(x) < 2, return an empty list
    """
    if len(x) < 2: #not enough x values
        return []
    
    #find most extreme for all lengths
    extremes = []
    for length in range(2, len(x)+1): 
        #generate extreme positive and negative for each length
        pos = find_extreme_trend(x, y, length, True)
        neg = find_extreme_trend(x, y, length, False)
        
        #if either has no slope value
        if pos is None:
            extremes.append(neg)
        elif neg is None:
            extremes.append(pos)
        
        #check if one is more extreme than other
        elif pos[2] - abs(neg[2]) > 1e-8:
            extremes.append(pos)
        elif abs(neg[2]) - pos[2] > 1e-8:
            extremes.append(neg)
            
        #check is equal, if equal, add the one with lowest index
        elif abs(pos[2]) - abs(neg[2]) <= 1e-8:
            if pos[0] < neg[0]:
                extremes.append(pos)
            else:
                extremes.append(neg)
    
    return extremes

def rmse(y, estimated):
    """
    Calculate the root mean square error term.

    Args:
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d numpy array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """
    num = 0
    
    #calculate sums for the numerator for all y values
    for value in range(len(y)):
        num += (y[value] - estimated[value])**2
    
    #compute rmse value
    rmse = (num/len(y))**.5
    
    return rmse

def evaluate_models_testing(x, y, models, display_graphs=False):
    """
    For each regression model, compute the RMSE for this model and if
    display_graphs is True, plot the test data along with the model's estimation.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        RMSE of your model evaluated on the given data points.

    RMSE should be rounded to 4 decimal places.

    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N test data sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N test data sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a numpy array storing the coefficients of
            a polynomial.
        display_graphs: A boolean whose value specifies if the graphs should be
            displayed

    Returns:
        A list holding the RMSE value for each model
    """
    total_rmse = []
    
    #generate RMSE for each model
    for model in range(len(models)):
        pred_y = np.polyval(models[model], x) 
        error = rmse(y, pred_y) #compute rsme
        total_rmse.append(error)
        
        #make a different graph for each model if specified
        if display_graphs == True:
            plt.plot(x,y, "bo") #blue dot for individual points
            plt.plot(x, pred_y, "r-") #red line for model
            plt.xlabel("Years")
            plt.ylabel("Temperature (Celcius)")
            plt.title("Fit of degree " + str(len(models[model])-1) + ", RMSE = " + str(round(error, 4)))
            plt.show()
                        
    return total_rmse 

if __name__ == '__main__':
    pass
    ##################################################################################
    # Problem 4A: DAILY TEMPERATURE
    data_set = Dataset("data.csv")
    x= list(range(1961, 2017))
    y = []
    [y.append(data_set.get_temp_on_date("SAN FRANCISCO", 12, 25, year)) for year in x]
         
     
    models = generate_models(x, y, [1])
    evaluate_models(x,y, models, display_graphs = True)
    
    ##################################################################################
    # Problem 4B: ANNUAL TEMPERATURE
    data_set = Dataset("data.csv")
    x = list(range(1961, 2017))
    y = data_set.get_yearly_averages(["SAN FRANCISCO"], x)
    
    models = generate_models(x, y, [1])
    evaluate_models(x, y, models, display_graphs = True)
    
    #4.1 - evaluating the yearly average seems to have a much better fit than a specific day
    #4.2 - noisy because they cover a lot of variation (all 12 months vs different years - both are broad)

    ##################################################################################
    # Problem 5B: INCREASING TRENDS
    data_set = Dataset("data.csv")
    x = list(range(1961, 2017))
    y = data_set.get_yearly_averages(["TAMPA"], x)
    
    trend = find_extreme_trend(x, y, 30, True)
    print("trend = ", trend)
    new_x = x[trend[0]: trend[1]+1]
    print("years = ", new_x)
    new_y = data_set.get_yearly_averages(["TAMPA"], new_x)
    
    models = generate_models(new_x, new_y, [1])
    evaluate_models(new_x, new_y, models, display_graphs = True)
    
    #5.1 - start year is 1962, end year is 1992, slope was .046
    ##################################################################################
    # Problem 5C: DECREASING TRENDS
    data_set = Dataset("data.csv")
    x = list(range(1961, 2017))
    y = data_set.get_yearly_averages(["TAMPA"], x)
    
    trend = find_extreme_trend(x, y, 15, False)
    print("decr. trend = ", trend)
    new_x = x[trend[0]: trend[1]+1]
    print("decr. years = ", new_x)
    new_y = data_set.get_yearly_averages(["TAMPA"], new_x)
    
    models = generate_models(new_x, new_y, [1])
    evaluate_models(new_x, new_y, models, display_graphs = True)
    
    #5.2 - start year is 1970, end year is 1985, slope is -.032
    #5.3 - the R^2 for the decreasing temperatures is very low indicating a poor fit while the
        #R^2 for the increasing temperatures is higher and has a better fit
        #therefore the trend seems to be that temperatures are increasing over time
    ##################################################################################
    # Problem 5D: ALL EXTREME TRENDS
    data_set = Dataset("data.csv")
    x = list(range(1961, 2017))
    y = data_set.get_yearly_averages(["TAMPA"], x)
    trends = find_all_extreme_trends(x, y)
    num_pos = 0
    num_neg = 0
    for slope in trends:
        if slope[2] > 0:
            num_pos += 1
        elif slope[2] < 0:
            num_neg += 1
    print("trends = ", trends)
    print("------------------")
    print("Positive Slopes: ", num_pos, "\n Negative Slopes: ", num_neg)
    
    #5.4 - 54 intervals resulted in a more extreme positive slope while only 1 interval had a more extreme
        #negative slope
    #5.5 It is a convincing argument that there are so many more positive slopes, but"Turn Down the AC" can
        #point to the slopes and say that they are not that strong (most under 1) 
    ##################################################################################
    # Problem 6B: PREDICTING
    data_set = Dataset("data.csv")
    x_train = TRAINING_INTERVAL
    y_train = data_set.get_yearly_averages(CITIES, x_train)
    
    models_train = generate_models(x_train, y_train, [2, 10])
    evaluate_models(x_train, y_train, models_train, display_graphs = True)
    
    # 6.1 - these models are very close in terms of R^2. The model of degree 10 has a slightly higher
        # R^2 (.01) and therefore suggests a better fit
        # However, I think the degree 10 model may be overfitting because it seems to respond to small 
        # changes in the training data
    x_test = TESTING_INTERVAL
    y_test = data_set.get_yearly_averages(CITIES, x_test)
    evaluate_models_testing(x_test, y_test, models_train, display_graphs = True)
    
    #6.4 - The model of degree 2 performed best because it has a much lower RMSE than the model of degree 10
        #This is different than the training models where degree 10 was a better fit. I think this is 
        #because the degree 10 was overfit to the training data
        
        
    y_san_train = data_set.get_yearly_averages(["SAN FRANCISCO"], x_train)
    models_san_train = generate_models(x_train, y_san_train, [2, 10])
    evaluate_models_testing(x_test, y_test, models_san_train, display_graphs = True)
    
    #6.5 - If we only used San Francisco, prediction results for the national average would be much lower
    #than the true values. The fot of degree 2 would still be better than the degree 10 which would again
    #overfit the data. 
    ##################################################################################
