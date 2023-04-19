import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#物理环境下的主机的属性
CPU_PROPERTIES_SMALL = [10, 9, 8, 7, 6, 5, 4, 3, 3, 3]#10个主机（小），每个主机上的 CPU的数量
CPU_PROPERTIES_LARGE = [10, 9, 8, 7, 6, 5, 4, 3, 3, 3, 10, 9, 8, 7, 6, 5, 4, 3, 3, 3]#20个主机（大），CPU的数量

LINK_PROPERTIES_BW_SMALL = [1000, 800, 600, 400, 400, 200, 200, 100, 100, 100]
LINK_PROPERTIES_BW_LARGE = [1000, 800, 600, 400, 400, 200, 200, 100, 100, 100, 1000, 800, 600, 400, 400, 200, 200, 100, 100, 100]

LINK_PROPERTIES_LAT_SMALL = [30, 50, 10, 20, 10, 30, 40, 50, 40, 20]
LINK_PROPERTIES_LAT_LARGE = [30, 50, 10, 20, 10, 30, 40, 50, 40, 20, 30, 50, 10, 20, 10, 30, 40, 50, 40, 20]

#SFC的特征---每个VNF对CPU数量、带宽数量、本身的处理时间
VNFD_PROPERTIES_SIZE_SMALL = [0, 4, 3, 3, 2, 2, 2, 1, 1]# 0 是一个占位，CPU的数量
VNFD_PROPERTIES_BW_SMALL =   [0, 100, 80, 60, 40, 20, 10, 50, 30]#带宽 需求
VNFD_PROPERTIES_LAT_SMALL =  [0, 100, 80, 60, 40, 20, 10, 50, 30]#处理过程的时延


class Environment(object):
    """
        Implementation of a sequence-to-sequence model based on dynamic multi-cell RNNs

        Attributes:
            num_cpus(int)                           -- Number of hosts
            num_vnfds(int)                          -- Number of VNF descriptors
            env_profile(str)                        -- Environment profile
            dict_vnf_profile(str)                   -- VNF dictionary profile
    """

    def __init__(self, num_cpus, num_vnfds, omigas, env_profile="small_default", dict_vnf_profile="small_default"):

        # Environment properties
        self.num_cpus = num_cpus
        self.num_vnfds = num_vnfds
        self.cpu_properties = [{"numSlots": 0} for _ in range(num_cpus)]
        self.link_properties = [{"bandwidth": 0, "latency": 0} for _ in range(num_cpus)]
        self.vnfd_properties = [{"size": 0, "bandwidth": 0, "latency": 0} for _ in range(num_vnfds + 1)]
        self.p_min = 200 #主机的开机的能耗（无VNF部署时）
        self.p_slot = 100#主机在有VNF运行时，CPU利用相关的能耗

        # Assign environmental properties   分大小不同的规模取出主机的属性与SFC的特征
        self._getEnvProperties(num_cpus, env_profile)
        self._getVnfdProperties(num_vnfds, dict_vnf_profile)

        # Environment cell slots
        self.max_slots = max([cpu["numSlots"] for cpu in self.cpu_properties])#取cpu_properties 列表的（键值对）值的最大
        self.cells = np.empty((self.num_cpus, self.max_slots))#随机初始化一个矩阵num_cpus:主机的数量，max_slots：最多的CPU

        #omigas的列数，即目标的数量
        self.objectives = np.zeros(len(omigas))
        #Initialize Environment variables
        self._initEnv()#环境的属性与SFC的属性不能被覆盖

    def _initEnv(self):

        # Clear environment
        self.cells[:] = np.nan #不是一个数字， 一个二维表，VNF占用CPU实际分配的表
        self.cpu_used = np.zeros(self.num_cpus)
        self.cpu_uti = np.zeros(self.num_cpus)#cpu利用率
        self.link_used = np.zeros(self.num_cpus)
        self.link_uti = np.zeros(self.num_cpus)#带宽利用率

        # Clear placement
        self.service_length = 0
        self.network_service = None
        self.placement = None
        self.first_slots = None #某个VNF可能占用多个CPU，firSt为占用CPU（主机上的）的开始位置
        self.reward = None
        self.constraint_occupancy = None
        self.constraint_bandwidth = None
        self.constraint_latency = None
        self.invalid_placement = False
        self.invalid_bandwidth = False
        self.invalid_latency = False

        self.link_latency = 0
        self.cpu_latency = 0

        self.objectives[:] = np.nan


    def _getEnvProperties(self,num_cpus, env_profile):

        if env_profile == "small_default":

            assert num_cpus == len(CPU_PROPERTIES_SMALL)

            for i in range(num_cpus):

                self.cpu_properties[i]["numSlots"] = CPU_PROPERTIES_SMALL[i] #[{"numSlots":CPU_PROPERTIES_SMALL[0]},{"numSlots":CPU_PROPERTIES_SMALL[1]，...}
                self.link_properties[i]["bandwidth"] = LINK_PROPERTIES_BW_SMALL[i]
                self.link_properties[i]["latency"] = LINK_PROPERTIES_LAT_SMALL[i]

        elif env_profile == "large_default":

            assert num_cpus == len(CPU_PROPERTIES_LARGE)

            for i in range(num_cpus):

                self.cpu_properties[i]["numSlots"] = CPU_PROPERTIES_LARGE[i]
                self.link_properties[i]["bandwidth"] = LINK_PROPERTIES_BW_LARGE[i]
                self.link_properties[i]["latency"] = LINK_PROPERTIES_LAT_LARGE[i]

        else:
            raise Exception('Environment not detected.')


    def _getVnfdProperties(self, num_vnfds, dict_vnf_profile):

        if dict_vnf_profile == "small_default":

            assert num_vnfds + 1 == len(VNFD_PROPERTIES_SIZE_SMALL)

            for i in range(num_vnfds + 1):

                self.vnfd_properties[i]["size"] = VNFD_PROPERTIES_SIZE_SMALL[i]
                self.vnfd_properties[i]["bandwidth"] = VNFD_PROPERTIES_BW_SMALL[i]
                self.vnfd_properties[i]["latency"] = VNFD_PROPERTIES_LAT_SMALL[i]

        else:
            raise Exception('VNF dictionary not detected.')


    def _placeSlot(self, cpu, vnf):
        """ Place VM """#放入主机的上的CPU，CPU被编号

        occupied_slot = np.nan

        for slot in range(self.cpu_properties[cpu]["numSlots"]):
            if np.isnan(self.cells[cpu][slot]):
                self.cells[cpu][slot] = vnf
                occupied_slot = slot
                break

        return occupied_slot


    def _placeVNF(self, i, cpu, vnf):
        """ Place VNF """

        if self.vnfd_properties[vnf]["size"] <= (self.cpu_properties[cpu]["numSlots"] - self.cpu_used[cpu]):

            for slot in range(self.vnfd_properties[vnf]["size"]):
                occupied_slot = self._placeSlot(cpu, vnf)

                # Anotate first slot used by the VNF
                if slot == 0:
                    self.first_slots[i] = occupied_slot

            self.cpu_used[cpu] += self.vnfd_properties[vnf]["size"]

        else:

            self.cpu_used[cpu] += self.vnfd_properties[vnf]["size"]
            self.first_slots[i] = -1

    def _computeLink(self):
        """ Compute link usage and link latency """

        self.bandwidth = max([self.vnfd_properties[vnf]["bandwidth"] for vnf in self.network_service])

        for i in range(self.service_length):

            cpu = self.placement[i]

            if i == 0:
                #self.link_used[cpu] += self.bandwidth
                self.link_used[cpu] += self.vnfd_properties[self.network_service[i]]["bandwidth"]
                self.link_latency += self.link_properties[cpu]["latency"]
            elif cpu != self.placement[i-1]: #相邻两个VNF不消耗link
                #self.link_used[cpu] += self.bandwidth
                self.link_used[cpu] += self.vnfd_properties[self.network_service[i]]["bandwidth"]
                self.link_latency += self.link_properties[cpu]["latency"]

            if i == self.service_length - 1:
                #self.link_used[cpu] += self.bandwidth
                self.link_used[cpu] += self.vnfd_properties[self.network_service[i]]["bandwidth"]
                self.link_latency += self.link_properties[cpu]["latency"]
            '''
            elif cpu != self.placement[i+1]:
                #self.link_used[cpu] += self.bandwidth
                self.link_used[cpu] += self.vnfd_properties[i]["bandwidth"]
                self.link_latency += self.link_properties[cpu]["latency"]
            '''

    def _computeReward(self, omiga):
        """ Compute reward signals """

        # Check occupancy
        self.constraint_occupancy = 0
        for i in range(self.num_cpus): #num_cpus = 主机的数量
            if self.cpu_used[i] > self.cpu_properties[i]["numSlots"]:
                self.invalid_placement = True  #当出现实际的CPU放置超过主机的属性时，放置是不合理
                self.constraint_occupancy += self.cpu_used[i] - self.cpu_properties[i]["numSlots"]

            self.cpu_uti[i] = self.cpu_used[i] / self.cpu_properties[i]["numSlots"]
            if self.cpu_uti[i] > 1:
                self.cpu_uti[i] = 1

        # Check bandwidth
        self.constraint_bandwidth = 0
        for i in range(self.num_cpus):
            if self.link_used[i] > self.link_properties[i]["bandwidth"]:
                self.invalid_bandwidth = True #当出现实际的link放置超过主机的属性时，放置是不合理
                self.constraint_bandwidth += self.link_used[i] - self.link_properties[i]["bandwidth"]

            self.link_uti[i] = self.link_used[i] / self.link_properties[i]["bandwidth"]
            if self.link_uti[i] > 1:
                self.link_uti[i] = 1

        # Check latency   VNF的处理时延（CPU时延） SFC上所有VNF的时延之和，即CPU时延由VNF的类型决定
        self.cpu_latency = sum([self.vnfd_properties[vnf]["latency"] for vnf in self.network_service[:self.service_length]])

        self.constraint_latency = 0
        if self.link_latency > self.cpu_latency:
            self.invalid_latency = True
            self.constraint_latency += self.link_latency - self.cpu_latency

        #所有主机的CPU的能耗
        powerConsume = 0
        for cpu in range(self.num_cpus):
            if self.cpu_used[cpu]:
                powerConsume += self.p_min + self.p_slot * self.cpu_used[cpu]
        #所有VNF消耗的带宽
        linkConsume = sum(self.link_used)

        #所有nodeCPU利用率的平均值
        #cpuUti = np.average(self.cpu_uti)
        cpuUti = sum(self.cpu_uti)/len(np.nonzero(self.cpu_uti)[0])

        # 所有nodeCPU利用率的平均值
        linkUti = sum(self.link_uti)/len(np.nonzero(self.link_uti)[0])

        # 定义不同的目标,第一个目标是能耗、第二个目标是CPU利用率（即拥塞率）
        self.objectives[0] = powerConsume/10

        #self.objectives[1] = linkConsume
        #self.objectives[1] = self.link_latency
        self.objectives[1] = cpuUti*100

        #第三个目标：链路利用率
        if len(self.objectives) == 3 :
            self.objectives[2] = linkUti*100

        self.reward = sum(np.multiply(omiga, self.objectives))


    def step(self, length, network_service, placement,omiga):
        """ Place network service """
        # network_service = SFC  placement = VNF放置的位置 length = SFC中VNF的数量，  每次放一个SFC

        self.service_length = length  #SFC中VNF的数量
        self.network_service = network_service #SFC
        self.placement = placement
        self.first_slots = -np.ones(length, dtype='int32')

        for i in range(length):
            self._placeVNF(i, placement[i], network_service[i])#依次将SFC的每个VNF放入placement中

        self._computeLink()
        self._computeReward(omiga)

    def clear(self):

        # Reset environmental variables
        self._initEnv()

    def render(self):
        """ Render environment using MatplotLib """#放置的结果可视化

        # Creates just a figure and only one subplot
        fig, ax = plt.subplots()
        ax.set_title('Environment')

        margin = 3
        margin_ext = 6
        xlim = 100
        ylim = 80

        # Set drawing limits
        plt.xlim(0, xlim)
        plt.ylim(-ylim, 0)

        # Set hight and width for the box
        high = np.floor((ylim - 2 * margin_ext - margin * (self.num_cpus - 1)) / self.num_cpus)
        wide = np.floor((xlim - 2 * margin_ext - margin * (self.max_slots - 1)) / self.max_slots)

        plt.text(1, 1, "Energy: {}".format(self.reward), ha="center", family='sans-serif', size=8)
        plt.text(10, 1, "Cstr occ: {}".format(self.constraint_occupancy), ha="center", family='sans-serif', size=8)
        plt.text(20, 1, "Cstr bw: {}".format(self.constraint_bandwidth), ha="center", family='sans-serif', size=8)
        plt.text(30, 1, "Cstr lat: {}".format(self.constraint_latency), ha="center", family='sans-serif', size=8)


        # Plot slot labels
        for slot in range(self.max_slots):
            x = wide * slot + slot * margin + margin_ext
            plt.text(x + 0.5 * wide, -3, "slot{}".format(slot), ha="center", family='sans-serif', size=8)

        # Plot cpu labels & placement empty boxes
        for cpu in range(self.num_cpus):
            y = -high * (cpu + 1) - (cpu) * margin - margin_ext
            plt.text(0, y + 0.5 * high, "cpu{}".format(cpu), ha="center", family='sans-serif', size=8)

            for slot in range(self.cpu_properties[cpu]["numSlots"]):
                x = wide * slot + slot * margin + margin_ext
                rectangle = mpatches.Rectangle((x, y), wide, high, linewidth=1, edgecolor='black', facecolor='none')
                ax.add_patch(rectangle)

        # Select service_length colors from a colormap
        cmap = plt.cm.get_cmap('hot')
        colormap = [cmap(np.float32(i+1)/(self.service_length+1)) for i in range(self.service_length)]

        # Plot service boxes
        for idx in range(self.service_length):
            vnf = self.network_service[idx]
            cpu = self.placement[idx]
            first_slot = self.first_slots[idx]

            for k in range(self.vnfd_properties[vnf]["size"]):

                # Plot ONLY if it is a valid placement
                if first_slot != -1:
                    slot = first_slot + k
                    x = wide * slot + slot * margin + margin_ext
                    y = -high * (cpu + 1) - cpu * margin - margin_ext
                    rectangle = mpatches.Rectangle((x, y), wide, high, linewidth=0, facecolor=colormap[idx], alpha=.9)
                    ax.add_patch(rectangle)
                    plt.text(x + 0.5 * wide, y + 0.5 * high, "vnf{}".format(vnf), ha="center", family='sans-serif', size=8)

        plt.axis('off')
        plt.show()


if __name__ == "__main__":

    # Define environment
    num_cpus = 10
    num_vnfds = 8

    env = Environment(num_cpus, num_vnfds, omigas=[1.0, 1.0, 1.0])

    # Allocate service in the environment
    service_length = 8
    network_service = [4, 8, 1, 4, 3, 6, 6, 8]
    #placement =       [2, 3, 2, 1, 1, 0, 0, 0]
    placement =       [2, 0, 2, 1, 1, 0, 0, 0]

    env.step(service_length, network_service, placement, omiga=[0.5, 0.5, 0.5])

    print("Placement Invalid: ", env.invalid_placement)
    print("Link used: ", env.link_used, "Invalid: ", env.invalid_bandwidth)
    print("CPU used:", env.cpu_used)
    print("CPU Latency: ", env.cpu_latency, "Link Latency: ", env.link_latency, "Invalid: ", env.invalid_latency)
    print("Reward: ", env.reward)
    for i in range(len(env.objectives)):
        print("objective_", i, env.objectives[i],  "   ")

    print("Constraint_occupancy: ", env.constraint_occupancy)
    print("Constraint_bandwidth: ", env.constraint_bandwidth)
    print("CPU utilization: ", env.cpu_uti, "CPU utilization average", np.average(env.cpu_uti))
    print("Link utilization: ", env.link_uti,"Link utilization average: ", np.average(env.link_uti))
    print("Constraint_latency: ", env.constraint_latency)

    env.render()
    env.clear()




