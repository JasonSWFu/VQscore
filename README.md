## Self-Supervised Speech Quality Estimation and Enhancement Using Only Clean Speech (ICLR 2024)
#### Szu-Wei Fu, Kuo-Hsuan Hung, Yu Tsao, Yu-Chiang Frank Wang

<center><img src="https://github.com/JasonSWFu/VQscore/blob/main/VQScore.png" width="800"></center>

## Training
To Train our speech enhancement (using only Clean Speech). Below is an example command.
```shell
python trainVQVAE.py -c config/SE_cbook_2048_1_128_2Transformer_vq_3_kernel_size_91.yaml --tag SE_cbook_2048_1_128_2Transformer_vq_3_kernel_size_91
```
To Train our speech Quality Estimatior (using only Clean Speech), VQScore. Below is an example command.
```shell
python trainVQVAE.py -c config/QE_cbook_size_2048_1_32_IN_input_encoder_z_Librispeech_clean.yaml --tag QE_cbook_size_2048_1_32_IN_input_encoder_z_Librispeech_clean
```

## Inference
Below is an example command for generating enhanced speech / estimated quality scores from the model.
Where '-c' is the path of the config file, '-m' is the path of the pretrained model, and '-i' is the path of input wav file.

```shell
python inference.py \
-c ./config/SE_cbook_2048_1_128_2Transformer_vq_3_kernel_size_91_2.yaml \
-m ./exp/SE_cbook_2048_1_128_2Transformer_vq_3_kernel_size_91_2/checkpoint-dnsmos_ovr=2.698_AT.pkl \
-i ./noisy_p232_005.wav
```
```shell
python inference.py \
-c ./config/QE_cbook_size_2048_1_32_IN_input_encoder_z_Librispeech_clean.yaml \
-m ./exp/QE_cbook_size_2048_1_32_IN_input_encoder_z_Librispeech_clean/checkpoint-dnsmos_ovr_CC=0.835.pkl \
-i ./noisy_p232_005.wav
```



## Pretrained Models
We provide the checkpoints of trained models in the corresponding ./exp/config_name folder.

* Note that the provided checkpoints are the results after we reorganize the code, so the results are slightly different from those shown in the paper.
* However, the overall trend should be similar.


## Collaboration
I'm open to collaboration! If you find this Self-Supervised SE/QE topic interesting, please let me know (e-mail: szuweif@nvidia.com). 

### Citation
If you find the code useful in your research, please cite our ICLR paper:
    
## References
* [vector-quantize](https://github.com/lucidrains/vector-quantize-pytorch) (for VQ-VAE)
