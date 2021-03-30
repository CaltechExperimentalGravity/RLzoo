#from scipy.signal import csd, psd

from scipy import signal
import numpy as np

def take_tf(input, output, samp_f):
    #print(samp_f)
    csd=signal.csd(input, output, samp_f)
    psd=signal.welch(input,samp_f)
    #print(csd, type(csd))
    #print(psd, type(psd))
    tf=np.array([])
    f_num=psd[0]
    csd_for=csd[1]
    psd_for=psd[1]
    for i in range(len(psd[0])):
        tf= np.append(tf, (csd_for[i])/(psd_for[i]))
    return [f_num, tf]
