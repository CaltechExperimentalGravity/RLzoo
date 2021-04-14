#from scipy.signal import csd, psd

from scipy import signal
import numpy as np

def take_tf(input, output, samp_f, f=-1):
    '''
    params:
    input: time series of the input signal
    output: time series of the output signal
    samp_f: sampling frequency in Hz
    '''
    csd=signal.csd(input, output, samp_f)
    psd=signal.welch(input,samp_f)
    tf=np.array([])
    f_num=psd[0]
    csd_for=csd[1]
    psd_for=psd[1]
    for i in range(len(psd[0])):
        tf = np.append(tf, (csd_for[i])/(psd_for[i]))
    if f<0:
        return [f_num, tf]
    else:
        return [
    
from matplotlib.mlab import psd, csd
def tfe(input, output, samp_f):
    print(len(input))
    print(len(output))
    """estimate transfer function from x to y, see csd for calling convention"""
    c_csd=csd(output, input, Fs=samp_f)
    c_psd=psd(input, Fs=samp_f)
    
    data_out=[]
    for i in range(length(c_csd)):
        data_out.append(c_csd[i]/c_psd[i])
    return data_out

