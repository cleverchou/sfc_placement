import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygmo as gmo

from mpl_toolkits.mplot3d import Axes3D

#from pygmo.util import *

class MultiObj:
    def __init__(self, objects, n):
        objects = np.array(objects)
        if n >= 0:
            self.objctive = objects[:, n]
        else:
            self.objctive = objects
        self.batch = n

        self.objN = len(self.objctive[0])

        self.pf = []
        self._get_paretoFront()
        self.pf = np.array(self.pf)

    def _getUtopiafor(self):
        '''对于最小化问题，目标空间每维的最小值的组合为理想点---2维'''
        m, n = len(self.objctive), len(self.objctive[0])
        return [min(self.objctive[i][j] for i in range(m)) for j in range(n)]


    def selectPre(self):
        '''找到目标空间距离理想点最近的点，返回该点及所在目标空间的位置'''
        utopia = self._getUtopiafor()

        #a = [self.objctive[i] - utopia[i] for i in range(0, len(utopia))]
        #a2 = [a[i] * a[i] for i in range(0, len(a))]
        minDis = 10e6
        pre, preIdex, i = [], 0, 0
        utopia_ = np.array(utopia)
        for ver in self.objctive:
            ver_ = np.array(ver)
            #两个向量之间的欧式距离
            distance = np.sqrt(np.sum(np.square(ver_ - utopia_)))
            if distance <= minDis:
                minDis = distance
                pre = ver[:]
                preIdex = i
            i += 1
        return pre, preIdex

    def showPF(self):
        # 必要输入
        title = 'PF_4_batch_'+ str(self.batch)  # 标题


        if self.objN == 2:
            plt.figure()  # 初始化一张图
            xlabel = 'f1'  # 横坐标标题
            ylabel = 'f2'  # 纵坐标标题
            f1 = self.pf[:, 0]
            f2 = self.pf[:, 1]
            plt.scatter(f1, f2)  # plot连线图,若要散点图将此句改为：plt.scatter(x,y) #散点图
            plt.grid(alpha=0.5, linestyle='-.')  # 网格线，更好看
            plt.title(title, fontsize=14)  # 画总标题 fontsize为字体，下同
            plt.xlabel(xlabel, fontsize=10)  # 画横坐标
            plt.ylabel(ylabel, fontsize=10)  # 画纵坐标
            plt.savefig("PF/" + title + '.pdf')
        else:
            fig = plt.figure()
            ax = plt.axes(projection='3d')

            f1 = self.pf[:, 0]
            f2 = self.pf[:, 1]
            f3 = self.pf[:, 2]
            ax.scatter3D(f1, f2, f3)

            plt.grid(alpha=0.5, linestyle='-.')  # 网格线，更好看
            plt.title(title, fontsize=14)
            plt.legend()
            # save figure
            plt.savefig("PF/" + title + '.pdf')



    def get_hyperVolumeFor2(self, refer_point=[0, 0]):
        '''
        计算超体积值
        :param solutions list 非支配解集，是解的目标函数值列表，形式如：[[object1,object2],[object1,object2],...]
        :param refer_point list 参考点，要被solutions内所有点支配，默认为[1.2,1.2]
        '''
        refer_point = np.array(refer_point)
        #solutions = np.array(solutions)
        solutions = sorted(self.pf, key=(lambda x:x[0]))  # 按第一个目标降序排序
        volume = 0
        for i in solutions:
            volume += abs(i[0] - refer_point[0]) * abs(i[1] - refer_point[1])
            #refer_point[0] -= refer_point[0] - i[0]
            refer_point[1] = refer_point[1] - i[1]
        return volume

    def _is_pareto_efficient(self):
        """
        Find the pareto-efficient points
        :param costs: An (n_points, n_costs) array
        :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
        """
        is_efficient = np.ones(self.objctive.shape[0], dtype=bool)
        for i, c in enumerate(self.objctive):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(self.objctive[is_efficient] <= c, axis=1)  # Keep any point with a lower cost
                is_efficient[i] = True  # And keep self
        return is_efficient

    def _get_paretoFront(self):
        ispf = self._is_pareto_efficient()
        #pf = []

        for i in range(len(self.objctive)):
            if ispf[i]:
                if len(self.pf) == 0:
                    self.pf.append(self.objctive[i])
                else:
                    append = True
                    for j in range(len(self.pf)):
                        if all(self.objctive[i] == self.pf[j]):
                            append = False
                            break
                    if append:
                        self.pf.append(self.objctive[i])
        if self.objN == 2:
            name = ['f1', 'f2']
        else:
            name = ['f1', 'f2', 'f3']
        pfall = pd.DataFrame(columns=name, data=self.pf)
        pfall.to_csv('PF/pfAll.csv', mode='a', encoding='gbk')
        #print(self.pf)
        #return self.pf

    def get_hyperVolume(self, refer_point=[0, 0]):
        #self.get_paretoFront()
        hv = gmo.hypervolume(self.pf)
        refer_point = np.array(refer_point)

        return hv.compute(refer_point)


if __name__ == "__main__":
    #objects = [[[0,1,], [1,1],[1,0]],[[0,2],[2,2],[0,1]],[[0,2],[2,2],[1,1]],[[0,2],[2,2],[0,0]]]
    objects = [[[0, 1, 1], [1, 1,1], [1, 3,3]], [[0, 2,1], [2, 2,0], [4, 1,4]], [[0, 2,1], [2, 2,1], [5,5,1]], [[0, 2,1], [2, 2,1], [2, 2,2]]]
    batchN = 2
    mob = MultiObj(objects, batchN)
    print("pf=",mob.pf)
    pre, prei = mob.selectPre()
    print("preference: ", "obj=", pre, "index=", prei, "objects=", objects[prei][batchN])
    #hv = mob.get_hyperVolumeFor2([2,2,2])
    #print("hv=", hv)
    hv = mob.get_hyperVolume([10,10,10])
    print("hv1=", hv)

    mob.showPF()