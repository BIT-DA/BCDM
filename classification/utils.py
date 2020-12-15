import numpy as np
import torch
from numpy.random import *
from torch.autograd import Variable
from torch.nn.modules.batchnorm import BatchNorm2d, BatchNorm1d, BatchNorm3d


def textread(path):
    # if not os.path.exists(path):
    #     print path, 'does not exist.'
    #     return False
    f = open(path)
    lines = f.readlines()
    f.close()
    for i in range(len(lines)):
        lines[i] = lines[i].replace('\n', '')
    return lines


def adjust_learning_rate(optimizer, epoch,iter_num,iter_per_epoch=4762,lr=0.001):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    ratio=(epoch*iter_per_epoch+iter_num)/(10*iter_per_epoch)
    lr=lr*(1+10*ratio)**(-0.75)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)


def cdd(output_t1,output_t2):
    mul = output_t1.transpose(0, 1).mm(output_t2)
    cdd_loss = torch.sum(mul) - torch.trace(mul)
    return cdd_loss


def get_theta(embedding_dim, num_samples=50):
    theta = [w/np.sqrt((w**2).sum())
             for w in np.random.normal(size=(num_samples, embedding_dim))]
    theta = np.asarray(theta)
    return torch.from_numpy(theta).type(torch.FloatTensor).cuda()


def sliced_wasserstein_distance(source_z, target_z,embed_dim, num_projections=256, p=1):
    # theta is vector represents the projection directoin
    batch_size = target_z.size(0)
    theta = get_theta(embed_dim, num_projections)
    proj_target = target_z.matmul(theta.transpose(0, 1))
    proj_source = source_z.matmul(theta.transpose(0, 1))

    w_distance = torch.abs(torch.sort(proj_target.transpose(0, 1), dim=1)[0]-torch.sort(proj_source.transpose(0, 1), dim=1)[0])
    w_distance=torch.mean(w_distance)
    # calculate by the definition of p-Wasserstein distance
    w_distance_p = torch.pow(w_distance, p)

    return w_distance_p.mean()
