import tensorflow as tf
import tensorlayer as tl

from common import math_utils
from common.value_networks import *
from common.policy_networks import *
from gym import spaces


def atari(env):
    alg_params = dict(
        gamma=0.9,
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 32  # dimension of hidden layers for the networks
        with tf.name_scope('AC'):
            with tf.name_scope('Critic'):
                critic = ValueNetwork(env.observation_space, hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Actor'):
                actor = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                hidden_dim_list=num_hidden_layer * [hidden_dim],
                                                output_activation=tf.nn.tanh)
        net_list = [actor, critic]
        alg_params['net_list'] = net_list
    if alg_params.get('optimizers_list') is None:
        a_lr, c_lr = 1e-4, 1e-2  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
        a_optimizer = tf.optimizers.Adam(a_lr)
        c_optimizer = tf.optimizers.Adam(c_lr)
        optimizers_list = [a_optimizer, c_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        seed=2,
        max_steps=200
    )

    return alg_params, learn_params


def classic_control(env):
    alg_params = dict(
        gamma=0.9,
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 32  # dimension of hidden layers for the networks
        with tf.name_scope('AC'):
            with tf.name_scope('Critic'):
                critic = ValueNetwork(env.observation_space, hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Actor'):
                actor = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                hidden_dim_list=num_hidden_layer * [hidden_dim],
                                                output_activation=tf.nn.tanh)
        net_list = [actor, critic]
        alg_params['net_list'] = net_list
    if alg_params.get('optimizers_list') is None:
        a_lr, c_lr = 1e-4, 1e-2  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
        a_optimizer = tf.optimizers.Adam(a_lr)
        c_optimizer = tf.optimizers.Adam(c_lr)
        optimizers_list = [a_optimizer, c_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        seed=2,
        max_steps=200
    )

    return alg_params, learn_params


def box2d(env):
    alg_params = dict(
        gamma=0.9,
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 32  # dimension of hidden layers for the networks
        with tf.name_scope('AC'):
            with tf.name_scope('Critic'):
                critic = ValueNetwork(env.observation_space, hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Actor'):
                actor = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                hidden_dim_list=num_hidden_layer * [hidden_dim],
                                                output_activation=tf.nn.tanh)
        net_list = [actor, critic]
        alg_params['net_list'] = net_list
    if alg_params.get('optimizers_list') is None:
        a_lr, c_lr = 1e-4, 1e-2  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
        a_optimizer = tf.optimizers.Adam(a_lr)
        c_optimizer = tf.optimizers.Adam(c_lr)
        optimizers_list = [a_optimizer, c_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        seed=2,
        max_steps=200
    )

    return alg_params, learn_params


def mujoco(env):
    alg_params = dict(
        gamma=0.9,
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 32  # dimension of hidden layers for the networks
        with tf.name_scope('AC'):
            with tf.name_scope('Critic'):
                critic = ValueNetwork(env.observation_space, hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Actor'):
                actor = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                hidden_dim_list=num_hidden_layer * [hidden_dim],
                                                output_activation=tf.nn.tanh)
        net_list = [actor, critic]
        alg_params['net_list'] = net_list
    if alg_params.get('optimizers_list') is None:
        a_lr, c_lr = 1e-4, 1e-2  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
        a_optimizer = tf.optimizers.Adam(a_lr)
        c_optimizer = tf.optimizers.Adam(c_lr)
        optimizers_list = [a_optimizer, c_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        seed=2,
        max_steps=200
    )

    return alg_params, learn_params


def robotics(env):
    alg_params = dict(
        gamma=0.9,
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 32  # dimension of hidden layers for the networks
        with tf.name_scope('AC'):
            with tf.name_scope('Critic'):
                critic = ValueNetwork(env.observation_space, hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Actor'):
                actor = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                hidden_dim_list=num_hidden_layer * [hidden_dim],
                                                output_activation=tf.nn.tanh)
        net_list = [actor, critic]
        alg_params['net_list'] = net_list
    if alg_params.get('optimizers_list') is None:
        a_lr, c_lr = 1e-4, 1e-2  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
        a_optimizer = tf.optimizers.Adam(a_lr)
        c_optimizer = tf.optimizers.Adam(c_lr)
        optimizers_list = [a_optimizer, c_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        seed=2,
        max_steps=200
    )

    return alg_params, learn_params


def rlbench(env):
    alg_params = dict(
        gamma=0.9,
    )
    if alg_params.get('net_list') is None:
        num_hidden_layer = 1  # number of hidden layers for the networks
        hidden_dim = 32  # dimension of hidden layers for the networks
        with tf.name_scope('AC'):
            with tf.name_scope('Critic'):
                critic = ValueNetwork(env.observation_space, hidden_dim_list=num_hidden_layer * [hidden_dim])
            with tf.name_scope('Actor'):
                actor = StochasticPolicyNetwork(env.observation_space, env.action_space,
                                                hidden_dim_list=num_hidden_layer * [hidden_dim],
                                                output_activation=tf.nn.tanh)
        net_list = [actor, critic]
        alg_params['net_list'] = net_list
    if alg_params.get('optimizers_list') is None:
        a_lr, c_lr = 1e-4, 1e-2  # a_lr: learning rate of the actor; c_lr: learning rate of the critic
        a_optimizer = tf.optimizers.Adam(a_lr)
        c_optimizer = tf.optimizers.Adam(c_lr)
        optimizers_list = [a_optimizer, c_optimizer]
        alg_params['optimizers_list'] = optimizers_list

    learn_params = dict(
        seed=2,
        max_steps=200
    )

    return alg_params, learn_params
