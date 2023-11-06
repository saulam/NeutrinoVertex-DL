# Datasets and model checkpoints

Please use the following Zenodo link to download all the necessary files for this repository: [Zenodo - doi/10.5281/zenodo.10075198](https://zenodo.org/doi/10.5281/zenodo.10075198).

## Datasets

The repository relies on several datasets, each of which is available as a compressed file. Please download the following datasets:

- `dataset_p.tar.gz`: proton dataset.
- `dataset_mu.tar.gz`: muon dataset.
- `dataset_D+.tar.gz`: deuterium dataset.
- `dataset_T+.tar.gz`: tritium dataset.

After downloading, you can extract the datasets using the following command:

```shell
tar -xvf dataset_*.tar.gz
```

And place them in the "datasets" folder located at the root directory of this repository. Make sure that the dataset folder structure aligns with the expected format.

Also, ensure that you extract and process the files within the file "metadata.tar.gz" in the same way to acquire additional information necessary for working with the datasets.

Make sure to configure the paths to these datasets accurately within the "config" folder at the root directory of this repository.

## Checkpoints

The "checkpoints.tar.gz" file contains pre-trained model weights for various models used in this project. To access these weights, extract the contents of the archive:

```shell
tar -xvf checkpoints.tar.gz
```

This will result in a folder with the following model weight files:

- `transformer_conf1_weights.pth`: trained weights for the decomposing transformer (Configuration 1).
- `transformer_conf2_weights.pth`: trained weights for the decomposing transformer (Configuration 2).
- `gan_p_conf1_weights.pth`: trained weights for the proton generator (Configuration 1).
- `gan_mu_conf1_weights.pth`: trained weights for the muon generator (Configuration 1).
- `gan_p_conf2_weights.pth`: trained weights for the proton generator (Configuration 2).
- `gan_mu_conf2_weights.pth`: trained weights for the muon generator (Configuration 2).
- `gan_D_conf2_weights.pth`: trained weights for the deuterium generator (Configuration 2).
- `gan_T_conf2_weights.pth`: trained weights for the tritium generator (Configuration 2).

Make sure to configure the paths to these model weight files accurately within the "config" folder at the root directory of this repository to ensure proper model loading.