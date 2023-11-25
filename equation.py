import numpy as np
import torch



class Equation(object):
    """Base class for defining PDE related function."""

    def __init__(self, eqn_config):   #传入字典描述方程
        self.dim = eqn_config["dim"]
        self.total_time = eqn_config["total_time"]
        self.num_time_interval = eqn_config["num_time_interval"]
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.y_init = None

    def sample(self, num_sample):
        """Sample forward SDE."""
        raise NotImplementedError

    def f_tf(self, t, x, y, z):
        """Generator function in the PDE."""
        raise NotImplementedError

    def g_tf(self, t, x):
        """Terminal condition of the PDE."""
        raise NotImplementedError


class HJBLQ(Equation):
    """HJB equation in PNAS paper doi.org/10.1073/pnas.1718942115"""
    def __init__(self, eqn_config):
        super(HJBLQ, self).__init__(eqn_config)
        self.x_init = torch.zeros(self.dim)
        self.sigma = np.sqrt(2.0)
        self.lambd = 40.0

    def sample(self, num_sample):   #num_sample is batchsize
        dw_sample = torch.randn(num_sample, self.dim, self.num_time_interval) * self.sqrt_delta_t
        x_sample = torch.zeros(num_sample, self.dim, self.num_time_interval + 1)
        x_sample[:, :, 0] = torch.ones(num_sample, self.dim) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample      #dw_sample is bat*dim*N,x_sample is bat*dim*N+1

    def f_tf(self, t, x, y, z):
        return -self.lambd * torch.sum(torch.square(z), 1, keepdim=True) / 2

    def g_tf(self, t, x):
        return torch.log((1 + torch.sum(torch.square(x), 1, keepdim=True)) / 2)
