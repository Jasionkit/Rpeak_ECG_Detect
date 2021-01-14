from numpy import *
import numpy as np
from pywt import *

def decomp(ecg,myWavelet,size):
    cache = {} # (A,D) = (Low, High)
    data = np.empty(size,'float32')
    
    # part 1 
    sw1 = swt(ecg, myWavelet, level=3) #[(A3, D3),(A2, D2), (A1, D1)]
    cache['a'] = sw1[2][0]
    cache['d'] = sw1[2][1]
    cache['aa'] = sw1[1][0]
    cache['ad'] = sw1[1][1]
    
    # part 2
    sw2 = swt(cache['ad'],myWavelet,level = 1, start_level = 2) #[(A1,D1)]

    # part 3
    sw3 = swt(cache['d'],myWavelet, level = 2, start_level = 1) #[(A2, D2), (A1, D1)]
    cache['da'] = sw3[1][0]
    cache['dd'] = sw3[1][1]
    
    # part 4
    sw4 = swt(cache['dd'],myWavelet, level = 1, start_level = 2) #[(A1, D1)]
    
    data[:,0],data[:,1] = sw1[0][0],sw1[0][1]
    data[:,2],data[:,3] = sw2[0][0],sw2[0][1]
    data[:,4],data[:,5] = sw3[0][0],sw3[0][1]
    data[:,6],data[:,7] = sw4[0][0],sw4[0][1]

    return (data - mean(data, axis=0)) / std(data, axis=0)
