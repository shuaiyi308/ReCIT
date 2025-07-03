
# This code is modified from https://github.com/jakesnell/prototypical-networks

import torch
import torch.nn as nn
import numpy as np
from methods.meta_template import MetaTemplate
import torch.nn.functional as F

class ProtoNet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, tf_path=None):
        super(ProtoNet, self).__init__(model_func,  n_way, n_support, tf_path=tf_path)
        self.loss_fn = nn.CrossEntropyLoss()
        self.method = 'ProtoNet'

    def reset_modules(self):
        return

    def set_forward(self,x,is_feature=False):
        z_support, z_query = self.parse_feature(x,is_feature)
        z_support = z_support.contiguous()
        z_query = z_query.contiguous()
        
        
        #### cos similarity is better ####
        z_support = F.normalize(z_support, p=2, dim=-1)
        z_query = F.normalize(z_query, p=2, dim=-1)
        
        
        #### simplest transductive calibration ####
        # z_support: [cps, spc, d], z_query: [cps, qpc, d]
        r_query = z_query.view(self.n_way * self.n_query, -1) # [cps*qpc, d]
        r_support = z_support.view(self.n_way * self.n_support, -1) # [cps*spc, d]
        #r_all_data = torch.cat([r_support, r_query], dim=0) # [cps*spc+cps*qpc, d]

        r_all_data = r_support
        #r_all_data = r_query
        r_mean = r_all_data.mean(dim=0) # [d]
        r_std = r_all_data.std(dim=0) # [d]
        
        r_mean = r_mean.unsqueeze(0).unsqueeze(0); r_std = r_std.unsqueeze(0).unsqueeze(0) # [1, 1, d]
        
        z_support = z_support - r_mean
        z_query = z_query - r_mean
        
        #z_support = (z_support - r_mean) / (r_std + 1e-5)
        #z_query = (z_query - r_mean) / (r_std + 1e-5)

        z_support = F.normalize(z_support, p=2, dim=-1)
        z_query = F.normalize(z_query, p=2, dim=-1)
        
        ###########################################
        
        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1) #the shape of z is [n_data, n_dim]
        z_query = z_query.view(self.n_way * self.n_query, -1)
        
        dists = euclidean_dist(z_query, z_proto)
        # dists = cosine_dist(z_query, z_proto)

        scores = -dists
        return scores

    def get_distance(self,x,is_feature = False):
        z_support, z_query = self.parse_feature(x,is_feature)
        z_support = z_support.contiguous()
        z_proto = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        z_query = z_query.contiguous().view(self.n_way* self.n_query, -1)
        return euclidean_dist(z_proto, z_proto)[0, :5].cpu().numpy()

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query))
        y_query = y_query.cuda()
        scores = self.set_forward(x)
        loss = self.loss_fn(scores, y_query)
        return scores, loss
    
def cosine_dist(x, y):
    # x: N x D
    # y: M x D
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    return 1 - torch.mm(x, y.t())


def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
    #return torch.pow(F.normalize(x, p=2, dim=-1) - F.normalize(y, p=2, dim=-1), 2).sum(2)


    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)