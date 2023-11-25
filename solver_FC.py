import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


import time
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt



DELTA_CLIP = 50.0


class BSDESolver(object):
    """The fully connected neural network model."""
    def __init__(self, config, bsde):    #传入字典config和equation类bdse
        self.eqn_config = config["eqn_config"]
        self.net_config = config["net_config"]
        self.bsde = bsde

        self.model = NonsharedModel(config, bsde)   #model
        self.y_init = self.model.y_init           #
  #      lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
   #         self.net_config.lr_boundaries, self.net_config.lr_values)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        #optimizer_1 = torch.optim.Adam(net_1.parameters(), lr=initial_lr)
        #scheduler_1 = toLambdaLR(optimizer_1, lr_lambda=lambda epoch: 1 / (epoch + 1))
    def train(self):
        start_time = time.time()
        training_history = []
        training_time=[]

        # begin sgd iteration
        for step in range(self.net_config["num_iterations"]+1):
            valid_data = self.bsde.sample(self.net_config["valid_size"])
            loss = self.loss_fn(valid_data,True)
            if step % self.net_config["logging_frequency"] == 0:

                y_init = self.model.y_init
                elapsed_time = time.time() - start_time
                training_history.append(y_init.item())
                training_time.append(step)
                if self.net_config["verbose"]:
                    print("step: %d,    loss: %.4f, Y0: %.4f,   elapsed time: %3f" % (
                        step, loss.data.item(), float(y_init), elapsed_time))
                    #将history数据加载到列表中，没有打印
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        plt.plot(np.array(training_time),np.array(training_history))
        plt.show()
#            self.train_step(self.bsde.sample(self.net_config.batch_size))
            #训练函数，返回training_history列表
        return np.array(training_history)

    def loss_fn(self, inputs, training):
        dw, x = inputs
        y_terminal = self.model(inputs, training)
        delta = y_terminal - self.bsde.g_tf(self.bsde.total_time, x[:, :, -1])
        # use linear approximation outside the clipped range
        loss = torch.sum(torch.where(torch.abs(delta) < DELTA_CLIP, torch.square(delta),
                                       2 * DELTA_CLIP * torch.abs(delta) - DELTA_CLIP ** 2))
         #一种奇怪的loss函数
        return loss

#    def grad(self, inputs, training):
 #       with tf.GradientTape(persistent=True) as tape:
  #          loss = self.loss_fn(inputs, training)
   #     grad = tape.gradient(loss, self.model.trainable_variables)
    #    del tape
     #   return grad

  #  @tf.function
   # def train_step(self, train_data):
    #    grad = self.grad(train_data, training=True)
     #   self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
     #这几个函数类似于torch中optimizer.step

class NonsharedModel(nn.Module):        #相当于nn.Module
    def __init__(self, config, bsde):
        super(NonsharedModel, self).__init__()
        self.eqn_config = config["eqn_config"]
        self.net_config = config["net_config"]
        self.bsde = bsde
        self.y_init = nn.Parameter(torch.from_numpy(np.random.uniform(low=self.net_config["y_init_range"][0],
                                                    high=self.net_config["y_init_range"][1],
                                                    size=[1])).float()
                                  )         #u0 类似于nn.parameter
        self.z_init = nn.Parameter(torch.from_numpy(np.random.uniform(low=-.1, high=.1,
                                                    size=[1, self.eqn_config["dim"]])).float()
                                  )         #delta_u0:1*N

        self.subnet = nn.ModuleList([FeedForwardSubNet(config) for _ in range(self.bsde.num_time_interval-1)]
                                    )   #建立N-1个自网络
    def forward(self, inputs, training):     #forward
        dw, x = inputs   #dw:bat*dim*N  x:bat*dim*N+1
        time_stamp = np.arange(0, self.eqn_config["num_time_interval"]) * self.bsde.delta_t
        #[delt,2delt....N-1 delt]
        all_one_vec = torch.ones(self.net_config["batch_size"],1)
        #这里函数调用太奇怪了
        #all_one_vec:bat*1
        y = all_one_vec * self.y_init
        #u0:batch*1
        z = torch.matmul(all_one_vec, self.z_init)
        #delta_u0:batch*N
        for t in range(0, self.bsde.num_time_interval-1):
            y = y - self.bsde.delta_t * (
                self.bsde.f_tf(time_stamp[t], x[:, :, t], y, z)
            ) + torch.sum(z * dw[:, :, t], 1, keepdim=True)
            z = self.subnet[t](x[:, :, t + 1], training) / self.bsde.dim
        # terminal time
        y = y - self.bsde.delta_t * self.bsde.f_tf(time_stamp[-1], x[:, :, -2], y, z) + \
            torch.sum(z * dw[:, :, -1], 1, keepdim=True)

        return y
        #batch*1


class FeedForwardSubNet(nn.Module):
    def __init__(self, config):
        super(FeedForwardSubNet, self).__init__()
        dim = config["eqn_config"]["dim"]
        num_hiddens = config["net_config"]["num_hiddens"]
                     #len(num_hiddens) + 2层batchnormalize
        self.bn_layers=nn.ModuleList([])
        self.bn_layers.append(nn.BatchNorm1d(
                num_features=dim,
                momentum=0.99,
                eps=1e-6,
            )
        )
        for k in range(0,len(num_hiddens)):
            self.bn_layers.append(nn.BatchNorm1d(num_features=num_hiddens[k],momentum=0.99,eps=1e-6,))

        self.bn_layers.append(nn.BatchNorm1d(
                num_features=dim,
                momentum=0.99,
                eps=1e-6,
            ))
#      self.dense_layers = nn.ModuleList([tf.keras.layers.Dense(num_hiddens[i],
 #                                                  use_bias=False,
  #                                                 activation=None)
   #                          for i in range(len(num_hiddens))])    #全连接层？'''
        self.dense_layers=nn.ModuleList([])
        self.dense_layers.append(nn.Linear(dim,num_hiddens[0]))
        for i in range(len(num_hiddens)-1):
            self.dense_layers.append(nn.Linear(num_hiddens[i],num_hiddens[i+1]))
        self.dense_layers.append(nn.Linear(num_hiddens[len(num_hiddens)-1],dim))
        # final output should be gradient of size dim
 #       self.dense_layers.append(tf.keras.layers.Dense(dim, activation=None))

    def forward(self, x, training):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        x = self.bn_layers[0](x)
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i+1](x)
            x = nn.functional.relu(x)
        x = self.dense_layers[-1](x)
        x = self.bn_layers[-1](x)
        return x             #每个自网络经多层全连接与batchnorm层
