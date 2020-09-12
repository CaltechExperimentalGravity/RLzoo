from rlzoo.common.env_wrappers import *
from rlzoo.common.utils import *
from rlzoo.algorithms import *
from gym import wrappers
import gym_tempcontrol
from time import time # just to have timestamps in the files

#import optuna
import neptune
#from neptunecontrib.monitoring.keras import NeptuneMonitor
neptune.init('ellabg/sandbox', api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiY2I1ODBhNzUtYzAzZC00MmMzLTgyOTktNTJkODY5YzY0MjljIn0=")
neptune.create_experiment(name='temp_ctrl')

import argparse
#import json

#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler, StandardScaler
#from keras.utils.vis_utils import plot_model, model_to_dot

# for getting data
#import sys, os

# thsi is how to best time a piece of code
from timeit import default_timer as timer

import pyvirtualdisplay

def dozoo(args):

    # EnvName = 'CartPole-v0'
    # EnvName = 'Pendulum-v0'
    
    EnvName = 'TempControl-v0'
    EnvType = 'classic_control'
    # EnvType = 'temp_ctrl'

    # env = build_env(EnvName, EnvType, state_type='vision')

    AlgName = 'SAC'
    env = build_env(EnvName, EnvType)
    #env = wrappers.Monitor(env, 'test_movie',force=True)
    alg_params, learn_params = call_default_params(env, EnvType, AlgName)

    # AlgName = 'DPPO'
    # number_workers = 2  # need to specify number of parallel workers in parallel algorithms like A3C and DPPO
    # env = build_env(EnvName, EnvType, nenv=number_workers)
    # alg_params, learn_params = call_default_params(env, EnvType, AlgName)
    # alg_params['method'] = 'clip'    # specify 'clip' or 'penalty' method for different version of PPO and DPPO
    # alg = eval(AlgName+'(**alg_params)')
    # alg.learn(env=env,  mode='train', render=False, **learn_params)
    # alg.learn(env=env,  mode='test', render=True, **learn_params)

    # AlgName = 'PPO'
    # env = build_env(EnvName, EnvType)
    # alg_params, learn_params = call_default_params(env, EnvType, AlgName)
    # alg_params['method'] = 'clip'    # specify 'clip' or 'penalty' method for different version of PPO and DPPO
    # alg = eval(AlgName+'(**alg_params)')
    # alg.learn(env=env,  mode='train', render=False, **learn_params)
    # alg.learn(env=env,  mode='test', render=True, **learn_params)

    #AlgName = 'A3C'
    #number_workers = 4  # need to specify number of parallel workers
    #env = build_env(EnvName, EnvType, nenv=number_workers)
    #alg_params, learn_params = call_default_params(env, EnvType, 'A3C')
    # alg = eval(AlgName+'(**alg_params)')
    # alg.learn(env=env,  mode='train', render=False, **learn_params)
    # alg.learn(env=env,  mode='test', render=True, **learn_params)

    display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()
    alg = eval(AlgName + '(**alg_params)')

    if args.train:
        learn_params['train_episodes'] = args.nepisodes
        learn_params['max_steps']      = args.max_steps
        alg.learn(env=env, mode='train', render=False, **learn_params)

    if args.test:
        # load trained model - this is so we can train on a headless machine an test locally
        alg.learn(env=env, mode='test', render=True, **learn_params)

    env.close()


if __name__ == '__main__':

    # parse some args here
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_steps', type=int, default=150,
                    help='How many time steps per episode')

    parser.add_argument('--nepisodes', type=int, default=100,
                    help='How many episode')

    parser.add_argument('--opt', type=str, default='Nadam',
                    help='Which Optimizer to use for training the network')

    parser.add_argument('--test',  default=False, action='store_true',
                    help='True = Load model and run to test the training.')

    parser.add_argument('--train', default=False, action='store_true',
                    help='True = Train the Model and Save results. False = no train.')

    parser.add_argument('--comment', type=str, default=None,
                    help='this is a comment string that gets added to the Neptune report')

    args = parser.parse_args()
    
    # Capture signals in the main thread
    #killer = bilinearHelper.GracefulKiller()


    dozoo(args)



