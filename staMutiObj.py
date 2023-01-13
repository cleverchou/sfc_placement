import numpy as np
from multi_objective import *
import os

def readCSV(fileName,objN):
    if objN == 2:
        data = np.loadtxt(fileName, delimiter=",", skiprows=0, usecols=[0, 1], dtype=np.float) # 2 objectives
    else:
        data = np.loadtxt(fileName, delimiter=",", skiprows=0, usecols=[0, 1, 2], dtype=np.float)
    #data = pd.read_csv(fileName, header=None, delimiter=",", engine='python')
    #data = np.float64(data)
    #shap = np.shape(data)
    #obj = data.reshape(1, shap[0], shap[1])
    return data

def getArPreference(batch_size, prei,constraint_bandwidth, constraint_latency, constraint_occupancy):
    acceptSFC = [1] * batch_size
    for batchN in range(batch_size):
        if constraint_bandwidth[prei[batchN]][batchN] > 0 or \
                constraint_latency[prei[batchN]][batchN] > 0 or \
                constraint_occupancy[prei[batchN]][batchN] > 0:
            acceptSFC[batchN] = 0

    return sum(acceptSFC) / batch_size

def getArGreed(objectN, batch_size,constraint_bandwidth,constraint_latency,constraint_occupancy):
    #objectN= len(omigas)
    acceptSFC = [0] * batch_size
    for batchN in range(batch_size):
        for i in range(objectN):
            if acceptSFC[batchN] == 1:
                break
            if constraint_bandwidth[i][batchN] == 0:
                for j in range(objectN):
                    if acceptSFC[batchN] == 1:
                        break
                    if constraint_latency[j][batchN] == 0:
                        for k in range(objectN):
                            if constraint_occupancy[k][batchN] == 0:
                                acceptSFC[batchN] = 1
                                break

    return sum(acceptSFC) / batch_size



if __name__ == "__main__":
    if os.path.isfile("PF/hvAll.csv"):
        os.remove("PF/hvAll.csv")
    objects = readCSV("PF/paretoSet.csv",3)
    mobj = MultiObj(objects, -1)
    pre, prei = mobj.selectPre()
    # PF
    mobj.showPF()
    if os.path.isfile("PF/paretoSet.csv"):
        os.remove("PF/paretoSet.csv")

    print("statics finished")




