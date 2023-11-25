{
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
    "num_iterations": 60000,
    "batch_size": 64,
    "valid_size": 256,

    "logging_frequency": 100,
    "dtype": "float64",
    "verbose": true
  }
}