#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 18:13:15 2020

@author: ella
"""

from gym.envs.registration import register
 
register(id='TempControl-v0', 
    entry_point='gym_tempcontrol.envs:TempEnv', 
)