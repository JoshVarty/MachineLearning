""" quiz materials for feature scaling clustering """

### FYI, the most straightforward implementation might 
### throw a divide-by-zero error, if the min and max
### values are the same
### but think about this for a second--that means that every
### data point has the same value for that feature!  
### why would you rescale it?  Or even use it at all?
import numpy

def featureScaling(data):
    #Probably want to check for min == max and throw error
    data = numpy.array(data, dtype=float)
    max = numpy.max(data)
    min = numpy.min(data)
    
    result = []

    for i in data:
        result.append((i-min)/(max - min))

    return result


# tests of your feature scaler--line below is input data
data = [115, 140, 175]
print featureScaling(data)
