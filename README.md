# DF-Iris CASIA-Thousand

## Install

```bash
python -m pip install -r requirements.txt
```

## Configure

Edit `configs/df_iris_casia_thousand.yaml` and set:

```yaml
data:
  data_root: "<CASIA-IrisV4-Thousand root>"
output:
  output_dir: "<output directory>"
model:
  pretrained_checkpoint: ""
```

## Train

```bash
python -m df_iris.train --config configs/df_iris_casia_thousand.yaml
```

## Evaluate

```bash
python -m df_iris.test eval-features --features <output directory>/features_test.npz
```

## Infer

```bash
python -m df_iris.test infer --checkpoint <output directory>/best.pt --input <image or directory> --output <features.npz>
```
