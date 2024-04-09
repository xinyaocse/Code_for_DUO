# DUO

This repository provides code of the work Stealthy and Efficient Adversarial Example Attack
on Video Retrieval Systems



**Demo:**

Here we provide two attack methods in the file "main":
>1. Pixel_Sparse_AE_attack is the codes of the pixel sparse AE attack method proposed by our previous work for the video retrieval system,
>1. Group_Sparse_AE_attack is the codes of the new group sparse AE attack method proposed by our current work to improve the query efficiency

```python
python main.py --s_net C3D --t_net I3D --gpus '0' --path 'Pixel_Sparse_AE_attack' --group 'P_spa' --dataset_name 'UCF101'&   # Pixel_Sparse_AE_attack
```

or

```python
python main.py --s_net C3D --t_net I3D --gpus '0' --path 'Group_Sparse_AE_attack' --group 'G_spa' --dataset_name 'UCF101'&   # Group_Sparse_AE_attack
```



