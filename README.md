# Pytorch_MultiGPU


## Single-GPU

```shell
python 1_trainer_SingleGPU.py
```

## Multi-GPU (DP)

```shell
python 2_trainer_DP.py
```

## Multi-GPU (DDP)


```shell
python -m torch.distributed.launch --nproc_per_node 4 3_trainer_DDP.py
```