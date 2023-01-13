#-*- coding: utf-8 -*-
"""
    Placement using Neural Combinational Optimization.

    Author: Ruben Solozabal, PhD student at the University of the Basque Country (UPV-EHU) - Bilbao
    Date: June 2019
"""
import logging
import tensorflow as tf
from environment import *
from service_batch_generator import *
from agent import *
from config import *
from solver import *
from tensorflow.python import debug as tf_debug
from tqdm import tqdm
import csv
import os
from first_fit import *
import copy

from generate_vector import *
from multi_objective import *
from staMutiObj import *

""" Globals """
DEBUG = True
OBJETIVE = 3


def print_trainable_parameters():
    """ Calculate the number of weights """

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        print('shape: ', shape, 'variable_parameters: ', variable_parameters)
        total_parameters += variable_parameters
    print('Total_parameters: ', total_parameters)
    #_ = input("input any key to continue")


def calculate_reward(env, networkServices, placement, num_samples, omiga):
    """ Evaluate the batch of states into the environmnet """#在一个批次（相当于batch_size个SFC请求）内，抽样N个放置结果，取其中最小的目标函数

    lagrangian = np.zeros(config.batch_size)
    penalty = np.zeros(config.batch_size)
    reward = np.zeros(config.batch_size)
    constraint_occupancy = np.zeros(config.batch_size)
    constraint_bandwidth = np.zeros(config.batch_size)
    constraint_latency = np.zeros(config.batch_size)
    objects = np.zeros((config.batch_size, OBJETIVE))

    reward_sampling = np.zeros(num_samples)
    objects_sampling = np.zeros((num_samples, OBJETIVE))
    constraint_occupancy_sampling = np.zeros(num_samples)
    constraint_bandwidth_sampling = np.zeros(num_samples)
    constraint_latency_sampling = np.zeros(num_samples)

    indices = np.zeros(config.batch_size)

    # Compute environment
    for batch in range(config.batch_size): #一个批次处理, 一批次相当于多个SFC部署完成，但每次只放置一个SFC，放置后后得到一个奖励
        for sample in range(num_samples):  #抽样，尝试了num_samples次，每次都是一个“干净”的环境
            env.clear()
            env.step(networkServices.service_length[batch], networkServices.state[batch], placement[sample][batch], omiga)#
            reward_sampling[sample] = env.reward
            #print(env.objectives)
            objects_sampling[sample] = env.objectives
            constraint_occupancy_sampling[sample] = env.constraint_occupancy
            constraint_bandwidth_sampling[sample] = env.constraint_bandwidth
            constraint_latency_sampling[sample] = env.constraint_latency

        penalty_sampling = agent.lambda_occupancy * constraint_occupancy_sampling + agent.lambda_bandwidth * constraint_bandwidth_sampling + agent.lambda_latency * constraint_latency_sampling
        lagrangian_sampling = reward_sampling + penalty_sampling  #奖励与惩罚两部分之和 lagrangian_sampling 是一个向量

        index = np.argmin(lagrangian_sampling) #取抽样后的优化目标的最小值对应的下标

        lagrangian[batch] = lagrangian_sampling[index]
        penalty[batch] = penalty_sampling[index]
        reward[batch] = reward_sampling[index]
        objects[batch] = objects_sampling[index]

        constraint_occupancy[batch] = constraint_occupancy_sampling[index]
        constraint_bandwidth[batch] = constraint_bandwidth_sampling[index]
        constraint_latency[batch] = constraint_latency_sampling[index]

        indices[batch] = index #index最小目标值的下标

    return lagrangian, penalty, reward, constraint_occupancy, constraint_bandwidth, constraint_latency, indices, objects

if __name__ == "__main__":

    """ Log """
    logging.basicConfig(level=logging.INFO)  # filename='example.log'
    # DEBUG, INFO, WARNING, ERROR, CRITICAL

    """ Configuration """
    config, _ = get_config()

    """ Environment """
    env = Environment(config.num_cpus, config.num_vnfd,  np.ones(OBJETIVE), config.env_profile)

    """ Network service generator """
    vocab_size = config.num_vnfd + 1
    networkServices = ServiceBatchGenerator(config.batch_size, config.min_length, config.max_length, vocab_size)

    """Generate multi objectives vectors"""
    multiObjVec = Mean_vector(20, OBJETIVE)

    """ Agent """
    agent = Agent(config)

    """使用3维模型推理2维，mark32为对应的维度"""
    mark32 = 2 # 2：第3个目标不记入


    """ Configure Saver to save & restore model variables """
    variables_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name]
    #saver = tf.train.Saver(var_list=variables_to_save, keep_checkpoint_every_n_hours=1.0)
    saver = tf.train.Saver(var_list=variables_to_save)
    print("Starting session ...")

    with tf.Session() as sess:

        # Activate Tensorflow CLI debugger
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        # Activate Tensorflow debugger in Tensorboard
        #sess = tf_debug.TensorBoardDebugWrapperSession(
        #    sess=sess,
        #    grpc_debug_server_addresses=['localhost:6064'],
        #    send_traceback_and_source_code=True)

        # Run initialize op
        sess.run(tf.global_variables_initializer())

        # Print total number of parameters
        print_trainable_parameters()

        # New batch of states
        networkServices.getNewState()

        # Mask
        mask = np.zeros((config.batch_size, config.max_length))
        for i in range(config.batch_size):
            for j in range(networkServices.service_length[i], config.max_length):
                mask[i, j] = 1

        omigas, _ = multiObjVec.get_similar_vectors()

        placement = []
        decoder_softmax = []
        lagrangian = []
        penalty = []
        reward = []
        constraint_occupancy = []
        constraint_bandwidth = []
        constraint_latency = []
        objects = []

        for objectN in range(len(omigas)):
                # 加载模型--->模型的编号为1
            #若第K维不为0，不进入推理中
            if omigas[objectN][mark32] != 0:
                continue
            #print("current omigas:", omigas[objectN])
            objectiveModel = "_".join(str(j) for j in omigas[objectN])
            saver.restore(sess, "{}_{}/".format(config.load_from, 1) + objectiveModel + "/model.ckpt")
            # saver.restore(sess, "{}_{}/tf_placement.ckpt".format(config.load_from, i+1))
            print("{}_{}/".format(config.load_from, 1) + objectiveModel + "/model.ckpt", " 's Model restored.")
            # print_trainable_parameters()

            # Compute placement
            feed = {agent.input_: networkServices.state,
                        agent.input_len_: [item for item in networkServices.service_length],
                        agent.mask: mask}

            _, placement_t, _, _ = \
                    sess.run([agent.actor.decoder_sampling, agent.actor.decoder_prediction, agent.actor.decoder_softmax_temp,
                              agent.actor.decoder_softmax], feed_dict=feed)

            # Interact with the environment with greedy placement---模拟部署
            lagrangian_t, penalty_t, reward_t, constraint_occupancy_t, constraint_bandwidth_t, constraint_latency_t, _, objects_t = \
                     calculate_reward(env, networkServices, placement_t, 1, omigas[objectN])

            # 保存每个目标的结果
            lagrangian.append(lagrangian_t)
            penalty.append(penalty_t)
            reward.append(reward_t)
            constraint_bandwidth.append(constraint_bandwidth_t)
            constraint_occupancy.append(constraint_occupancy_t)
            constraint_latency.append(constraint_latency_t)
            objects.append(objects_t[:, [0, 1]])
            placement.append(placement_t[0])

            print("finish placement : ", objectN, omigas[objectN])

        print("start  statistics ")
        hv = [0] * config.batch_size
        prei = [0] * config.batch_size
        for batchN in range(config.batch_size):
            mobj = MultiObj(objects, batchN)
            _, prei_t = mobj.selectPre()

            #preference
            prei[batchN] = prei_t

            # PF
            #mobj.showPF()

            # HV
            hv[batchN] = mobj.get_hyperVolume([600, 100])

        name = ['hv']
        hvall = pd.DataFrame(columns=name, data=hv)
        hvall.to_csv('PF/hvAll.csv', mode='a', encoding='gbk')

        #print("AR_bandwidth:", constraint_bandwidth)
        #print("AR_latency:", constraint_latency)
        #print("AR_occupancy:", constraint_occupancy)

        acceptSFC = getArPreference(config.batch_size, prei, \
                                        constraint_bandwidth, constraint_latency, constraint_occupancy)
        print("Accpet radio from the preference:", acceptSFC)
        #acceptSFC = getArGreed(len(omigas), config.batch_size, \
        #                                constraint_bandwidth, constraint_latency, constraint_occupancy)
        #print("Accpet radio from the greed:", acceptSFC)