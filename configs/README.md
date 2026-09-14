# Configs

Saved hyperparameters and random seeds for each model go here, for reproducibility.

Example structure:
```
configs/
  mlp_config.json
  cnn1d_config.json
  lstm_config.json
  tabnet_config.json
```

Each config file should include: random seed, learning rate, batch size, number of epochs, layer sizes/architecture details, and any other key hyperparameters used for the final reported results.