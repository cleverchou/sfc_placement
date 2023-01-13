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

        # Learn model
        if config.learn_mode:
            """
                Learning
            """

            # Restore model 本次实验不需要
            if config.load_model:
                saver.restore(sess, "{}/tf_placement.ckpt".format(config.load_from))
                print("\nModel restored.")


            # Summary writer
            writer = tf.summary.FileWriter("summary/repo", sess.graph)

            if config.save_model:
                filePath = "{}/learning_history.csv".format(config.save_to)

                if not os.path.exists(os.path.dirname(filePath)):
                    os.makedirs(os.path.dirname(filePath))

                if os.path.exists(filePath) and not config.load_model:
                    os.remove(filePath)

            print("\nStart learning...")

            try:
                omigas = multiObjVec.get_mean_vectors()
                for objectN in range(len(omigas)):
                    if objectN > 0:
                        #从已有的模型中选择上一个相邻模型，加载其参数
                        rObjModel = "_".join(str(j) for j in omigas[objectN-1])
                        saver.restore(sess, "{}/".format(config.save_to) + rObjModel + "/model.ckpt")

                    objectiveModel = "_".join(str(i) for i in omigas[objectN])

                    episode = 0
                    for episode in range(config.num_epoch):

                        # New batch of states   每次可以生成batch个SFC请求
                        networkServices.getNewState()

                        # Mask i为SFC的数量 j为每个SFC上VNF的数量,实际的SFC长度
                        mask = np.zeros((config.batch_size,config.max_length))
                        for i in range(config.batch_size):
                            for j in range(networkServices.service_length[i], config.max_length):
                                mask[i, j] = 1 #若0表示有效，1表示没有VNF

                        # RL Learning
                        feed = {agent.input_: networkServices.state,
                                agent.input_len_: [item for item in networkServices.service_length],
                                agent.mask: mask}

                        # Run placement
                        placement, decoder_softmax, _, baseline = sess.run([agent.actor.decoder_exploration, agent.actor.decoder_softmax, agent.actor.attention_plot, agent.valueEstimator.value_estimate], feed_dict=feed)
                        # positions, attention_plot = sess.run([agent.actor.positions, agent.actor.attention_plot], feed_dict=feed)

                        # Interact with the environment to return reward
                        lagrangian, penalty, reward, constraint_occupancy, constraint_bandwidth, constraint_latency, indices, objs = calculate_reward(env, networkServices, placement, 1, omigas[objectN])

                        placement_ = np.zeros((config.batch_size, config.max_length))
                        for batch in range(config.batch_size):
                            placement_[batch] = placement[int(indices[batch])][batch] #placement_抽样的最优解？

                        feed = {agent.placement_holder: placement_,
                                agent.input_: networkServices.state,
                                agent.input_len_: [item for item in networkServices.service_length],
                                agent.mask: mask,
                                agent.baseline_holder: baseline,
                                agent.lagrangian_holder: [item for item in lagrangian]}

                        # Update our value estimator 更新值的估算，基线函数的估算？
                        feed_dict_ve = {agent.input_: networkServices.state,
                                     agent.valueEstimator.target: lagrangian}

                        _, loss = sess.run([agent.valueEstimator.train_op, agent.valueEstimator.loss], feed_dict_ve)

                        # Update actor  更新策略？
                        summary, _, loss_rl = sess.run([agent.merged, agent.train_step, agent.loss_rl], feed_dict=feed)

                        # Print learning
                        if episode == 0 or episode % 100 == 0:

                            print("------------")
                            print("Episode: ", episode)
                            print("Minibatch loss: ", loss_rl)
                            print("Network service[batch0]: ", networkServices.state[0])
                            print("Network service[batch1]: ", networkServices.state[1])
                            print("Len[batch0]", networkServices.service_length[0])
                            print("Len[batch1]", networkServices.service_length[1])
                            print("Placement[batch0]: ", placement_[0])
                            print("Placement[batch1]: ", placement_[1])

                            # agent.actor.plot_attention(attention_plot[0])
                            # print("prob:", decoder_softmax[0][0])
                            # print("prob:", decoder_softmax[0][1])
                            # print("prob:", decoder_softmax[0][2])

                            print("Baseline[batch0]: ", baseline[0])
                            print("Reward[batch0]: ", reward[0])
                            print("Penalty[batch0]: ", penalty[0])
                            print("Lagrangian[batch0]: ", lagrangian[0])
                            print("Baseline[batch1]: ", baseline[1])
                            print("Reward[batch1]: ", reward[1])
                            print("Penalty[batch1]: ", penalty[1])
                            print("Lagrangian[batch1]: ", lagrangian[1])


                            print("Value Estimator loss: ", np.mean(loss))
                            print("Mean penalty: ", np.mean(penalty))
                            print("Count_nonzero: ", np.count_nonzero(penalty))

                        if episode % 5000 == 0:

                            # Save in summary
                            writer.add_summary(summary, episode)

                        if config.save_model and (episode == 0 or episode % 5000 == 0):

                            # Save in csv
                            csvData = ['omiga:  {}'.format(omigas[objectN]),
                                       'batch: {}'.format(episode),
                                       ' network_service[batch 0]: {}'.format(networkServices.state[0]),
                                       ' placement[batch 0]: {}'.format(placement_[0]),
                                       ' reward: {}'.format(np.mean(reward)),
                                       ' lagrangian: {}'.format(np.mean(lagrangian)),
                                       ' baseline: {}'.format(np.mean(baseline)),
                                       ' advantage: {}'.format(np.mean(lagrangian) - np.mean(baseline)),
                                       ' penalty: {}'.format(np.mean(penalty)),
                                       ' minibatch_loss: {}'.format(loss_rl)]

                            filePath = "{}/learning_history.csv".format(config.save_to)
                            with open(filePath, 'a') as csvFile:
                                writer2 = csv.writer(csvFile)
                                writer2.writerow(csvData)

                            csvFile.close()

                        # Save intermediary model variables
                        #if config.save_model and episode % max(1, int(config.num_epoch / 5)) == 0 and episode != 0:
                        #    save_path = saver.save(sess, "{}/tmp.ckpt".format(config.save_to), global_step=episode)
                        #    print("\n Intermediary Model saved in file: %s" % save_path)

                        episode += 1

                    print("\nLearning COMPLETED!")
                    #保存每个学习模型--保存后再修改文件夹，否则上一次的内容将被覆盖
                    #save_path = saver.save(sess, "{}/".format(config.save_to)+objectiveModel+"/model.ckpt")
                    save_path = saver.save(sess, "{}/".format(config.save_to) + "tmp" + "/model.ckpt")
                    os.rename("{}/".format(config.save_to) + "tmp", "{}/".format(config.save_to)+objectiveModel)
                    print("\nObjective Model is saved in file: %s" % objectiveModel)

            except KeyboardInterrupt:
                print("\nLearning interrupted by user.")

            # Save model
            if config.save_model:
                save_path = saver.save(sess, "{}/tf_placement.ckpt".format(config.save_to))
                print("\nFinal Model saved in file: %s" % save_path)

        else:
            """
                Inference
            """

            # New batch of states
            networkServices.getNewState()

            # Mask
            mask = np.zeros((config.batch_size, config.max_length))
            for i in range(config.batch_size):
                for j in range(networkServices.service_length[i], config.max_length):
                    mask[i, j] = 1

            omigas = multiObjVec.get_mean_vectors()

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
                # 加载模型--->模型的编号为2
                objectiveModel = "_".join(str(j) for j in omigas[objectN])
                saver.restore(sess, "{}_{}/".format(config.load_from, 2) + objectiveModel + "/model.ckpt")
                # saver.restore(sess, "{}_{}/tf_placement.ckpt".format(config.load_from, i+1))
                print("{}_{}/".format(config.load_from, 2) + objectiveModel + "/model.ckpt", " 's Model restored.")
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
                objects.append(objects_t)
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
                hv[batchN] = mobj.get_hyperVolume([600, 100, 100])  #
                #hv[batchN] = mobj.get_hyperVolume([600, 100])

            name = ['hv']
            hvall = pd.DataFrame(columns=name, data=hv)
            hvall.to_csv('PF/hvAll.csv', mode='a', encoding='gbk')

            #print("AR_bandwidth:", constraint_bandwidth)
            #print("AR_latency:", constraint_latency)
            #print("AR_occupancy:", constraint_occupancy)

            acceptSFC = getArPreference(config.batch_size, prei, \
                                        constraint_bandwidth, constraint_latency, constraint_occupancy)
            print("Accpet radio from the preference:", acceptSFC)
            acceptSFC = getArGreed(len(omigas), config.batch_size, \
                                        constraint_bandwidth, constraint_latency, constraint_occupancy)
            print("Accpet radio from the greed:", acceptSFC)