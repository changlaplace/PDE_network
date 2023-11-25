"""
The main file to run BSDE solver to solve parabolic partial differential equations (PDEs).

"""



import numpy as np


import equation as eqn
from solver_FC import BSDESolver
from matplotlib import pyplot as plt



def main():

    config={
        "eqn_config": {
        "_comment": "HJB equation in PNAS paper doi.org/10.1073/pnas.1718942115",
        "eqn_name": "HJBLQ",
        "total_time": 1.0,
        "dim": 100,
        "num_time_interval": 20
        },
        "net_config": {
        "y_init_range": [0, 1],
        "num_hiddens": [110, 110],
        "lr_values": [1e-2, 1e-2],
        "lr_boundaries": [1000],
        "num_iterations": 50000,
        "batch_size": 64,
        "valid_size": 64,
        "logging_frequency": 100,
        "dtype": "float64",
        "verbose": True
        }
    }

    bsde = eqn.HJBLQ(config["eqn_config"])



    bsde_solver = BSDESolver(config, bsde)
    training_history = bsde_solver.train()

#下面是存history，也不用管，我们直接打印出来就行了


if __name__ == '__main__':
    main()
