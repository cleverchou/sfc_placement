import numpy as np

class Mean_vector:
    # m维空间，H:[0/H,1/H.....H/H]
    def __init__(self, H=10, m=3):
        self.H = H
        self.m = m
        self.stepsize = 1 / H
        print("\nGenerating vector")

    def perm(self, sequence):
        # ！！！ 序列全排列，且无重复
        l = sequence
        if (len(l) <= 1):
            return [l]
        r = []
        for i in range(len(l)):
            if i != 0 and sequence[i - 1] == sequence[i]:
                continue
            else:
                s = l[:i] + l[i + 1:]
                p = self.perm(s)
                for x in p:
                    r.append(l[i:i + 1] + x)
        return r

    def get_mean_vectors(self):
        #print("\nget mean vector")
        #生成权均匀向量
        H = self.H
        m = self.m
        sequence = []
        for ii in range(H):
            sequence.append(0)
        for jj in range(m - 1):
            sequence.append(1)
        ws = []

        pe_seq = self.perm(sequence)
        for sq in pe_seq:
            s = -1
            weight = []
            for i in range(len(sq)):
                if sq[i] == 1:
                    w = i - s
                    w = (w - 1) / H
                    s = i
                    weight.append(w)
            nw = H + m - 1 - s
            nw = (nw - 1) / H
            weight.append(nw)
            if weight not in ws:
                ws.append(weight)
        return ws

    def get_sorted_vectors(self):
        #print("\nget sorted vector")
        #按0的个数排序
        mv = self.get_mean_vectors()

        oIndex = [0 for _ in range(self.m+1)]
        ws = [0 for _ in range(len(mv))]

        for i in range(len(mv)):
            count0 = mv[i].count(0)  #0的个数
            oIndex[self.m-count0] += 1 #下标对应目标的数量，子目标的数量为m-count0的个数
        for i in range(self.m):
            oIndex[i+1] += oIndex[i]
        oi = list(oIndex)

        for i in range(len(mv)):
            count0 = mv[i].count(0)  # 0的个数
            ws[oIndex[self.m-count0-1]] = mv[i]
            oIndex[self.m-count0-1] += 1

        return ws, oi

    def get_similar_vectors(self):
        #print("\nget similar vector")
        #按0的个数排序
        m_v_s, oi = self.get_sorted_vectors()
        mvidex = np.zeros(len(m_v_s))
        sim1 = list()
        sim2 = list()#如果目标的个数为2，用两个表记录w的相似关系，开始的w为0,1 和1，0
        sim3 = list()#如果目标的个数为3，开始的w为0,0,1 和0,1,0 以及1,0,0

        if self.m == 2:
            sim1.append(m_v_s[0])
            sim2.append(m_v_s[1])
            mvidex[0] = 1 #mvidex记录当前的w是否被加入相似关系的表中
            mvidex[1] = 1
            simp1 = 0 #记录当前w的位置
            simp2 = 0
            sim1n = 1

            while np.count_nonzero(mvidex) < len(m_v_s):
                cv = np.array(sim1[simp1])
                tMin1 = 999999
                tMinIdex1 = 0
                for si1 in range(oi[1],oi[2]):
                    if mvidex[si1] == 0:
                        t = np.sqrt(np.sum(np.square(cv - np.array(m_v_s[si1]))))
                        if t < tMin1:
                            tMin1 = t
                            tMinIdex1 = si1
                sim1.append(m_v_s[tMinIdex1])
                mvidex[tMinIdex1] = 1
                simp1 += 1

                if np.count_nonzero(mvidex) == len(m_v_s):
                    break
                else:
                    sim1n += 1

                cv = np.array(sim2[simp2])
                tMin2 = 999999
                tMinIdex2 = 0
                for si2 in range(oi[1], oi[2]):
                    if mvidex[si2] == 0:
                        t = np.sqrt(np.sum(np.square(cv - np.array(m_v_s[si2]))))
                        if t < tMin2:
                            tMin2 = t
                            tMinIdex2 = si2
                sim2.append(m_v_s[tMinIdex2])
                mvidex[tMinIdex2] = 1
                simp2 += 1

            mvsidex = np.zeros(len(m_v_s))#mvidex重新定义，用于记录不同相似W的开始位置
            mvsidex[0] = 1 #当mvsidex==1，使用随机方式初始化参数，在训练时
            mvsidex[sim1n+1] =1 #两个目标，所以有两个位置需要被记录，用于随机化
            return sim1 + sim2 + sim3, mvsidex

        else:  #目标的个数 == 3， 暂不考虑更多维
            sim1.append(m_v_s[0])
            sim2.append(m_v_s[1])
            sim3.append(m_v_s[2])
            mvidex[0] = 1  # mvidex记录当前的w是否被加入相似关系的表中
            mvidex[1] = 1
            mvidex[2] = 1

            simp1 = 0
            simp2 = 0
            simp3 = 0
            sim1n = 1
            sim2n = 1
            work_2zero = 1

            while np.count_nonzero(mvidex) < len(m_v_s):
                if work_2zero == 1:
                    cv = np.array(sim1[simp1])
                    tMin1 = 999999
                    tMinIdex1 = 0
                    for si1 in range(oi[1],oi[2]):
                        if mvidex[si1] == 0:
                            t = np.sqrt(np.sum(np.square(cv - np.array(m_v_s[si1]))))
                            if t < tMin1:
                                tMin1 = t
                                tMinIdex1 = si1
                    if tMin1 != 999999:
                        sim1.append(m_v_s[tMinIdex1])
                        mvidex[tMinIdex1] = 1
                        simp1 += 1
                        sim1n += 1

                    cv = np.array(sim2[simp2])
                    tMin2 = 999999
                    tMinIdex2 = 0
                    for si2 in range(oi[1], oi[2]):
                        if mvidex[si2] == 0:
                            t = np.sqrt(np.sum(np.square(cv - np.array(m_v_s[si2]))))
                            if t < tMin2:
                                tMin2 = t
                                tMinIdex2 = si2
                    if tMin2 != 999999:
                        sim2.append(m_v_s[tMinIdex2])
                        mvidex[tMinIdex2] = 1
                        simp2 += 1
                        sim2n += 1

                    cv = np.array(sim3[simp3])
                    tMin3 = 999999
                    tMinIdex3 = 0
                    for si3 in range(oi[1], oi[2]):
                        if mvidex[si3] == 0:
                            t = np.sqrt(np.sum(np.square(cv - np.array(m_v_s[si3]))))
                            if t < tMin3:
                                tMin3 = t
                                tMinIdex3 = si3
                    if tMin3 != 999999:
                        sim3.append(m_v_s[tMinIdex3])
                        mvidex[tMinIdex3] = 1
                        simp3 += 1

                    if np.count_nonzero(mvidex) == oi[2]:
                        work_2zero = 0
                else:
                    cv = np.array(sim1[simp1])
                    tMin1 = 999999
                    tMinIdex1 = 0
                    for si1 in range(oi[2], oi[3]):
                        if mvidex[si1] == 0:
                            t = np.sqrt(np.sum(np.square(cv - np.array(m_v_s[si1]))))
                            if t < tMin1:
                                tMin1 = t
                                tMinIdex1 = si1
                    if tMin1 != 999999:
                        sim1.append(m_v_s[tMinIdex1])
                        mvidex[tMinIdex1] = 1
                        simp1 += 1
                        sim1n += 1

                    cv = np.array(sim2[simp2])
                    tMin2 = 999999
                    tMinIdex2 = 0
                    for si2 in range(oi[2], oi[3]):
                        if mvidex[si2] == 0:
                            t = np.sqrt(np.sum(np.square(cv - np.array(m_v_s[si2]))))
                            if t < tMin2:
                                tMin2 = t
                                tMinIdex2 = si2
                    if tMin2 != 999999:
                        sim2.append(m_v_s[tMinIdex2])
                        mvidex[tMinIdex2] = 1
                        simp2 += 1
                        sim2n += 1

                    cv = np.array(sim3[simp3])
                    tMin3 = 999999
                    tMinIdex3 = 0
                    for si3 in range(oi[2], oi[3]):
                        if mvidex[si3] == 0:
                            t = np.sqrt(np.sum(np.square(cv - np.array(m_v_s[si3]))))
                            if t < tMin3:
                                tMin3 = t
                                tMinIdex3 = si3
                    if tMin3 != 999999:
                        sim3.append(m_v_s[tMinIdex3])
                        mvidex[tMinIdex3] = 1
                        simp3 += 1


            mvsidex = np.zeros(len(m_v_s))  # mvidex重新定义，用于记录不同相似W的开始位置
            mvsidex[0] = 1  # 当mvsidex==1，使用随机方式初始化参数，在训练时
            mvsidex[sim1n] = 1
            mvsidex[sim1n +sim2n ] = 1
            return sim1 + sim2 + sim3, mvsidex





    def save_mv_to_file(self, mv, name='out.csv'):
    #保存为csv
        f = np.array(mv, dtype=np.float32)
        #f = np.round(f, 2)
        np.savetxt(fname=name, X=f)

    def test(self):
    #测试
        m_v = self.get_mean_vectors()
        print("mv: ", m_v)

        m_v_s, oi = self.get_sorted_vectors()
        print("\nmvs: ", m_v_s,"\nmv_sort_index: ",oi)

        a, b = self.get_similar_vectors()
        print("\nmv_sim: ", a, "\nmv_sim_index: ", b)

        #self.save_mv_to_file(a, 'test.csv')

        #print(oi)

if __name__ == "__main__":
    mv = Mean_vector(20, 3)
    mv.test()

