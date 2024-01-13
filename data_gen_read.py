import numpy as np
import pandas as pd

## Generate artificial data
def generate_data(num_data = 500, step = 1, func = "Linear", pattern = 1):
    '''
    Generate artificial data with different patterns 

    function 1: linear
    function 2: sin(x)

    pattern 0: return data as it is
    pattern 1: add a mask every 5 numbers
    pattern 2: add a linear function
    '''
    ## generate the index from 0 to num_data
    idx = np.arange(0, num_data) 

    ## determine the function
    if (func == "Linear"):
        initial_data = idx
    elif (func == "Sin"):
        initial_data = np.sin(idx)
    
    ## determine the patterns
    if(pattern == 0):
        return initial_data
        
    for idx in range(0, len(initial_data)):
        if(idx % 50 == 0):
            if(pattern == 1):
                if(idx - 4 >= 0): initial_data[idx] += 1
                if(idx - 3 >= 0): initial_data[idx] += 2
                if(idx - 2 >= 0): initial_data[idx] += 3
                if(idx - 1 >= 0): initial_data[idx] += 4
                if(idx >= 0 and idx < len(initial_data)): initial_data[idx] += 5
                if(idx + 1 < len(initial_data)): initial_data[idx] += 4
                if(idx + 2 < len(initial_data)): initial_data[idx] += 3
                if(idx + 3 < len(initial_data)): initial_data[idx] += 2
                if(idx + 4 < len(initial_data)): initial_data[idx] += 1
                
            elif(pattern == 2):
                initial_data[idx] = initial_data[idx]*5 + 8

    return initial_data

def import_data(dataset = "EVcharging_data.csv"):
    '''
    Import existing data from directory
    '''
    data = pd.read_csv(dataset)
    return data.to_numpy()

def read_data(datatype = 1, num_data = 500, step = 1, func = "Linear", pattern = 1, dataset = "EVcharging_data_univariate_p.csv"):
    '''
    User can choose to generate data or import data
    1: generate data
    2: read data
    '''
    if(datatype == 1):
        return generate_data(num_data = 500, step = 1, func = "Linear", pattern = 1)
    else:
        return import_data(dataset = dataset)