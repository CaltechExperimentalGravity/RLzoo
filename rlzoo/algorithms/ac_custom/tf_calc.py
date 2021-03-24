#from scipy.signal import csd, psd

from scipy import signal
import numpy as np

def take_tf(input, output, samp_f):
    csd=signal.csd(input, output, samp_f)
    psd=signal.welch(input,samp_f)
    print(len(csd), type(csd))
    print(len(psd), type(psd))
    tf=np.array([])
    for i in range(len(psd)):
        np.append(tf, csd[i]/psd[i])
    return tf

    
def plot_tf(x,y):
   return
