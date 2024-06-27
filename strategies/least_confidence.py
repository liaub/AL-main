import numpy as np
import torch
import pickle
from .strategy import Strategy
# *最小置信度策略，选择最不确定的n个
class LeastConfidence(Strategy):
    def __init__(self, X, Y, Z, original_data, net, args):
        '''
        X:input data stream
        Y:dirft type
        Z:dirft point
        '''
        super(LeastConfidence, self).__init__(X, Y, Z, original_data, net, args)
        self.X = X
        self.Y = Y
        self.Z = Z
        self.original_data = original_data
        self.net = net
        self.args = args


    def query(self, BASE_PATH):
        types, probs, locs = self.predict_prob(self.X, BASE_PATH)
        U = probs[torch.where(probs < self.args.alpha)]
        D1 = self.original_data[torch.where(probs < self.args.alpha)]
        if len(U) < self.args.query_num:
            if len(U) == 1:
                D1 = np.expand_dims(D1, axis=0)
        else:
            D1 = D1[U.sort()[1][:self.args.query_num]]

        with open(self.args.DATA_FILE + '/point_statistic.pkl', 'rb') as f:
            point_ref = pickle.load(f)
        minimum_dist = []
        for loc in locs:
            minimum_dist.append(np.min(np.abs(int(loc.item()) - np.array(point_ref))))
        minimum_dist = np.argsort(np.array(minimum_dist))[::-1]
        D2 = self.original_data[minimum_dist[:self.args.query_num]]
        combined_array = np.concatenate((D1, D2), axis=0)
        # flattened_array = combined_array.flatten()
        unique_elements = np.unique(combined_array, axis=0)
        return unique_elements