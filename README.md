# TAEGAN

This repository contains the official implementation of the paper
**"TAEGAN: Revisit GANs for Tabular Data Generation"**, published at the
**17th Asian Conference on Machine Learning (ACML 2025)**, PMLR volume 304.
[[Paper]](https://proceedings.mlr.press/v304/li26c.html)

## Pre-requisites

1. `Python>=3.10`.
2. Python package `torch`, `tqdm`, `sklearn`, `numpy`.

## Execution

### Prepare Data

```shell
python run.py -o OUT_DIR prepare -s DATA_FILE
```

What the prepare step does is described in `run.py` under `prepare` function. 
Relevant code can be easily reconstructed from open-source CTAB-GAN+ implementation.
We decided not to expose our processing code because we are using code source from enterprise, which is not supposed to 
be exposed, The part of code actually called from this private code essentially does the same thing as CTAB-GAN+ data 
preprocessing, with only code structure and efficiency improvements.

### Train Model

```shell
python run.py -o OUT_DIR train -b BATCH_SIZE -e EPOCHS -w WARMUP_EPOCHS
```

The trained model weights are found in `OUT_DIR/generator.pt` and `OUT_DIR/discriminator.pt`.

### Sample Tensor

```shell
python run.py -o OUT_DIR sample -n N_ROWS -b BATCH_SIZE
```

The sampled tensor is found in `OUT_DIR/sampled.pt`. Note that the output of this step is still a transformed tensor 
with one-hot and VGM data encoding.

### Recover Sampled Data

```shell
python run.py -o OUT_DIR recover
```

This is the inverse step of `prepare`, with its description also found inside `run.py`, in function `recover`.
Since this is the inverse step of `prepare`, which is dependent on the data processing module, this part of code is also
not exposed, but would be very easy to reconstruct.
The output should then be saved as a csv file, which is the actually generated tabular data.

## Example

A running example is provided in `diabetes-demo` using the `diabetes` dataset from OpenML. 
Content inside is the outcome of `prepare` step.

One can directly run the following to get the sampled tensor:

```shell
python run.py -o diabetes-demo train
python run.py -o diabetes-demo sample -n 256
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@InProceedings{pmlr-v304-li26c,
  title = {TAEGAN: Revisit GANs for Tabular Data Generation},
  author = {Li, Jiayu and Zhao, Zilong and Yee, Kevin and Javaid, Uzair and Sikdar, Biplab},
  booktitle = {Proceedings of the 17th Asian Conference on Machine Learning},
  pages = {670--685},
  year = {2025},
  editor = {Lee, Hung-yi and Liu, Tongliang},
  volume = {304},
  series = {Proceedings of Machine Learning Research},
  month = {09--12 Dec},
  publisher = {PMLR},
  pdf = {https://raw.githubusercontent.com/mlresearch/v304/main/assets/li26c/li26c.pdf},
  url = {https://proceedings.mlr.press/v304/li26c.html}
}
```

## License

This code is released for non-commercial research and academic purposes only. See [`LICENSE`](LICENSE) for details.
