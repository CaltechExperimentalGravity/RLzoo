#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 13:29:02 2020

@author: ella

custom loss function to get ISE (integral square error)

ISE = int^(t)_(0) e^2 dt where e = (value - reference value)

"""

import tensorflow as tf

#trapezium method
def integral(y, x):
    dx = (x[-1] - x[0]) / (int(x.shape[0]) - 1)
    return ((y[0] + y[-1])/2 + tf.reduce_sum(y[1:-1])) * dx

def ISE_loss(y_actual, y_pred):
    ISE_loss = integral(tf.square(y_actual-y_pred), (y_actual-y_pred))
    return ISE_loss
