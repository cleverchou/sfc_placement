import numpy as np


class ServiceBatchGenerator(object):
    """
        Implementation of a random service chain generator   生成随机的SFC  每次生成多个SFC
         state[i,k] 第i个SFC包括最多k个VNF
         service_length每个SFC的实际VNF数量
        Attributes:
            state[batch_size, max_service_length] -- Batch of random service chains
            service_length[batch_size]            -- Array containing services length
    """
    def __init__(self, batch_size, min_service_length, max_service_length, vocab_size):
        """
        Args:
            batch_size(int)         -- Number of service chains to be generated   每次产生SFC的数量
            min_service_length(int) -- Minimum service length    每个SFC中最少有几个VNF
            max_service_length(int) -- Maximum service length    每个SFC中最多有几个 VNF
            vocab_size(int)         -- Size of the VNF dictionary   VNF的类型用数字代替
        """
        self.batch_size = batch_size
        self.min_service_length = min_service_length
        self.max_service_length = max_service_length
        self.vocab_size = vocab_size

        self.service_length = np.zeros(self.batch_size,  dtype='int32')
        self.state = np.zeros((self.batch_size, self.max_service_length),  dtype='int32')

    def getNewState(self):
        """ Generate new batch of service chain """

        # Clean attributes
        self.state = np.zeros((self.batch_size, self.max_service_length), dtype='int32')
        self.service_length = np.zeros(self.batch_size,  dtype='int32')

        # Compute random services
        for batch in range(self.batch_size):
            self.service_length[batch] = np.random.randint(self.min_service_length, self.max_service_length+1, dtype='int32')
            for i in range(self.service_length[batch]):
                vnf_id = np.random.randint(1, self.vocab_size,  dtype='int32')
                self.state[batch][i] = vnf_id


if __name__ == "__main__":

    # Define generator
    batch_size = 5
    min_service_length = 2
    max_service_length = 6
    vocab_size = 8

    env = ServiceBatchGenerator(batch_size, min_service_length, max_service_length, vocab_size)
    env.getNewState()
    print("ok")


