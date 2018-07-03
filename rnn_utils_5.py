import os
import numpy as np
import csv
import struct

def loadData(filePath):
    """Load the EEG data from the file
    Arguments: 
    filePath -- path where to cache the dataset locally

    Returns:
        Tuple of Numpy arrays: '(x_train, y_train), (x_test, y_test)'
            x_train -- [420, 23, 178]
            y_train -- [420, 23, 1]
            x_test  -- [80, 23, 178]
            y_test  -- [80, 23, 1]
    """

    #wifilePath = 'C:/Users/zheng/Dropbox/ima camp/data science/project/'
    EEGdata = os.path.join(filePath, 'EEGdata.csv') # the file path

    names = []
    X = []
    y = []

    with open(EEGdata) as data:
        csvReader = csv.reader(data)
        for row in csvReader:
        	X.append(row[1:179])
        	y.append(row[179])
        	names.append(row[0].split('.'))

    
    X = np.asarray(X[1:]).astype(np.int)
    y = np.asarray(y[1:]).astype(np.int)
    names = names[1:]


    sample = np.zeros((500,23)).astype(np.int)
    patient = np.zeros(500).astype(np.int)
    for k in range(11500):
    	name = names[k]
    	timestep = int(name[0].replace("X", "")) - 1
    	middle = int(name[1].replace("V",""))*1000
    	for i in range(500):
    		if len(name) == 3:
    		    if middle+int(name[2]) == patient[i]:
    			    sample[i,timestep] = k
    			    break
    		    elif patient[i] == 0:
    			    patient[i] = middle + int(name[2])
    			    sample[i,timestep] = k
    			    break
    		elif len(name) == 2:
    			if middle == patient[i]:
    			    sample[i,timestep] = k
    			    break
    			elif patient[i] == 0:
    			    patient[i] = middle
    			    sample[i,timestep] = k
    			    break

    X_train = np.zeros((420,23,178)).astype(np.int)
    y_train = np.zeros((420,23,1)).astype(np.int)
    X_test = np.zeros((80,23,178)).astype(np.int)
    y_test = np.zeros((80,23,1)).astype(np.int)


    for i in range(420):
    	for j in range(23):
    		X_train[i,j,:] = X[sample[i,j]]
    		y_train[i,j,0] = y[sample[i,j]]
    for i in range(420,500):
    	for j in range(23):
    		X_test[i-420,j,:] = X[sample[i,j]]
    		y_test[i-420,j,0] = y[sample[i,j]]

    X_train = np.asarray(X_train).astype(np.int)
    y_train = np.asarray(y_train).astype(np.int)
    X_test = np.asarray(X_test).astype(np.int)
    y_test = np.asarray(y_test).astype(np.int)


    return (X_train, y_train), (X_test, y_test)


def preprocessData(EEGData):
    """preprocess EEG data, scale the X, convert Y to 0 or 1
    Arguments:
        EEGData -- Tuple of Numpy arrays for EEG data: '(x_train_original, y_train_original), (x_test_original, y_test_original)'
            x_train_original -- [420, 23, 178]
            y_train_original -- [420, 23, 1]
            x_test_original  -- [80, 23, 178]
            y_test_original  -- [80, 23, 1]

    Returns:
        EEGData_preprocessed -- Tuple of Numpy arrays for EEG data that is preprocessed: '(x_train_preprocessed, y_train_preprocessed), (x_test_preprocessed, y_test_preprocessed)'
            x_train_preprocessed -- [420, 23, 178]
            y_train_preprocessed -- [420, 23, 5]
            x_test_preprocessed -- [80, 23, 178]
            y_test_preprocessed -- [80, 23, 5]
    """
    
    (x_train_original, y_train_original), (x_test_original, y_test_original) = EEGData

    x_train_preprocessed = x_train_original.astype('float32') / 330
    # convert to one-hot vector
    y_train_preprocessed = np.zeros((len(x_train_preprocessed), 23, 5))
    for i in range(len(x_train_original)):
    	for j in range(23):
            if y_train_original[i,j] == 1:
                y_train_preprocessed[i,j,0] = 1
            elif y_train_original[i,j] == 2:
                y_train_preprocessed[i,j,1] = 1
            elif y_train_original[i,j] == 3:
                y_train_preprocessed[i,j,2] = 1
            elif y_train_original[i,j] == 4:
                y_train_preprocessed[i,j,3] = 1
            elif y_train_original[i,j] == 5:
                y_train_preprocessed[i,j,4] = 1

    x_test_preprocessed = x_test_original.astype('float32') / 330
    # convert to one-hot vector
    y_test_preprocessed = np.zeros((len(x_test_preprocessed), 23, 5))
    for i in range(len(x_test_original)):
    	for j in range(23):
            if y_test_original[i,j] == 1:
                y_test_preprocessed[i,j,0] = 1
            elif y_test_original[i,j] == 2:
                y_test_preprocessed[i,j,1] = 1
            elif y_test_original[i,j] == 3:
                y_test_preprocessed[i,j,2] = 1
            elif y_test_original[i,j] == 4:
                y_test_preprocessed[i,j,3] = 1
            elif y_test_original[i,j] == 5:
                y_test_preprocessed[i,j,4] = 1



    return (x_train_preprocessed, y_train_preprocessed), (x_test_preprocessed, y_test_preprocessed)

def mini_batch(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a random minibatch from (X, Y)
    
    Arguments:
    X -- input data, of shape (number of examples, input size)
    Y -- true "label" vector of shape (number of examples, 10)
    mini_batch_size -- size of the mini-batch, integer
    
    Returns:
    mini_batch_X, mini_batch_Y
    """
    
    np.random.seed(seed)            
    m = len(X)                  # number of training examples
    # Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]

    # Create minibatch
    mini_batch_X = shuffled_X
    mini_batch_Y = shuffled_Y
    
    return (mini_batch_X, mini_batch_Y)

