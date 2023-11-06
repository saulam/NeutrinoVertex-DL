# Usage of training scripts


## Script 1: `train/train_transformer_conf1.py`

Training of the decomposing transformer using the first configuration, where each event consist of 1 muon and 1-5 protons:

To run `train_transformer_conf1.py`, open a terminal from the root project directory. Then, run the following command:

```bash
python -m train.train_transformer_conf1 [OPTIONS]
```


## Script 2: `train/train_transformer_conf1.py`

Training of the decomposing transformer using the first configuration, where each event consist of 1 muon, 0-4, protons, 0-1 deuterium, 0-1 tritium:

To run `train_transformer_conf2.py`, open a terminal from the root project directory. Then, run the following command:

```bash
python -m train.train_transformer_conf2 [OPTIONS]
```

**Input arguments**

Both scripts share the same input arguments:

| Argument | Description                          | Default value                          |
|---|--------------------------------------|----------------------------------------|
| `-cp`, `--config_path` | Path of configuration file           | config/decomposing_transformer_v1.json |
| `-hs`, `--hidden` | Hidden size of transformer model     | 192                                    |
| `-dr`, `--dropout` | Dropout of the model                 | 0.1                                    |
| `-el`, `--encoder_layers` | Number of encoder layers             | 10                                     |
| `-dl`, `--decoder_layers` | Number of decoder layers             | 10                                     |
| `-a`, `--attn_heads` | Number of attention heads            | 16                                     |
| `-b`, `--batch_size` | Batch size                           | 512                                    |
| `-e`, `--epochs` | Number of epochs                     | 12420                                  |
| `-w`, `--num_workers` | Dataloader worker size               | 32                                     |
| `--lr` | Learning rate of the optimizer       | 2e-3                                   |
| `-lrd`, `--lr_decay` | Learning rate decay of the scheduler | 0.9                                    |
| `-ag`, `--accum_grad_batches` | Batches for gradient accumulation    | 4                                      |
| `-st`, `--scheduler_steps` | Scheduler steps in one cycle         | 400                                    |
| `-wd`, `--weight_decay` | Weight decay of the optimiser        | 1e-2                                   |
| `-b1`, `--beta1` | Adam first beta value                | 0.9                                    |
| `-b2`, `--beta2` | Adam second beta value               | 0.999                                  |


## Script 3: `train/train_gan.py`

Training of the generative adversarial network (GAN) that learns to create particle images.

To run `train_gan.py`, open a terminal from the root project directory. Then, run the following command:

```bash
python -m train.train_gan [OPTIONS]
```

**Input arguments**

| Argument          | Description                                      | Default Value |
|-------------------|--------------------------------------------------|---------------|
| -cp, --config_path| Path of configuration file                       | config/gan_p_conf1.json |
| -is, --input_size | Input dimension (per cube)                      | 1             |
| -ls, --label_size | Number of labels (kinematic parameters)         | 6             |
| -ns, --noise_size | Size of the noise                                | 512           |
| -hs, --hidden     | Hidden size of transformer model                | 64            |
| -l, --layers      | Number of layers                                 | 2             |
| -a, --attn_heads  | Number of attention heads                        | 8             |
| -dr, --dropout    | Dropout of the model                             | 0.1           |
| -b, --batch_size  | Batch size                                       | 32            |
| -e, --epochs      | Number of epochs                                | 100           |
| -w, --num_workers | Dataloader worker size                           | 8             |
| --lr              | Learning rate of the optimiser                   | 5e-5          |
| -wd, --weight_decay | Weight decay of the optimiser                 | 0             |
| -cr, --crit_repeats | Critic iterations per generator               | 5             |
| -lgp, --lambda_gp | Lambda value for gradient penalty              | 10            |


To run the proton GAN training (configuration 1) with our parameters:

```bash
python -m train.train_gan -cp "config/gan_p_conf1.json" -ls 6
```

To run the muon GAN training (configuration 1) with our parameters:

```bash
python -m train.train_gan -cp "config/gan_mu_conf1.json" -ls 9
```

To run the proton GAN training (configuration 2) with our parameters:

```bash
python -m train.train_gan -cp "config/gan_p_conf2.json" -ls 6
```

To run the muon GAN training (configuration 2) with our parameters:

```bash
python -m train.train_gan -cp "config/gan_mu_conf2.json" -ls 9
```

To run the deuterium GAN training (configuration 2) with our parameters:

```bash
python -m train.train_gan -cp "config/gan_D_conf2.json" -ls 6
```

To run the tritium GAN training (configuration 2) with our parameters:

```bash
python -m train.train_gan -cp "config/gan_T_conf2.json" -ls 6
```

