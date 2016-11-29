#
#
# Regression and Classification programming exercises
#
#

#
#	In this exercise we will be taking a small data set and computing a linear function
#	that fits it, by hand.
#	

#	the data set

import numpy as np

sleep = [5,6,7,8,10]
scores = [65,51,75,75,86]


def compute_regression(sleep, scores):

    #	First, compute the average amount of each list
    avg_sleep = np.mean(sleep)
    avg_scores = np.mean(scores)

    #	Then normalize the lists by subtracting the mean value from each entry
    normalized_sleep = sleep - avg_sleep
    normalized_scores = scores - avg_scores

    #	Compute the slope of the line by taking the sum over each student
    #	of the product of their normalized sleep times their normalized test score.
    #	Then divide this by the sum of squares of the normalized sleep times.

    slope = np.sum((normalized_sleep * normalized_scores))/np.sum((normalized_sleep * normalized_sleep))

    #	Finally, We have a linear function of the form
    #	y - avg_y = slope * ( x - avg_x )
    #	y = slope * ( x - avg_x ) + avg_y
    #	y = slope * x - slope * avg_x  + avg_y 
    #	Rewrite this function in the form
    #	y = m * x + b
    #	Then return the values m, b

    # 
    m = slope
    b = (slope * avg_sleep + avg_scores)

    return m,b


if __name__=="__main__":
    m,b = compute_regression(sleep,scores)
    print "Your linear model is y={}*x+{}".format(m,b)