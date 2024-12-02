## Self-Supervised Speech Quality Estimation and Enhancement Using Only Clean Speech (ICLR 2024)
#### Szu-Wei Fu, Kuo-Hsuan Hung, Yu Tsao, Yu-Chiang Frank Wang

### Update 2024/12/2
Provide the "inference_folder" code to evaluate the utterances in a folder. We found that the inference speed of VQScore is quite fast (15.5 hours of speech takes less than 2 minutes on a single A100 GPU), making it suitable for filtering out noisy training data when training speech enhancement or TTS models.


### Introduction
This work is about training a speech quality estimator and enhancement model WITHOUT any labeled (paired) data. Specifically, during training, we only need CLEAN speech for model training.

<center><img src="https://github.com/JasonSWFu/VQscore/blob/main/VQScore.png" width="600"></center>

## Environment
CUDA Version: 12.2

python: 3.8

* Note: To use 'CUDAExecutionProvider' for accelerated DNSMOS ONNX model inference, please check CUDA and ONNX Runtime version compatibility, [here](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

## Dataset used in the paper/code
If you want to train from scratch, please download the dataset to the corresponding path depicted in the .csv and .pickle files.

Speech enhancement:

=> Training: [clean speech of VoiceBank-DEMAND trainset](https://datashare.ed.ac.uk/handle/10283/2791) (Its original sampling rate is 48kHz, you have to down-sample it to 16kHz)

=> validation: As in MetricGAN-U, [noisy speech (speakers p226 and p287) of VoiceBank-DEMAND trainset](https://datashare.ed.ac.uk/handle/10283/2791)

=> Evaluation: [noisy speech of VoiceBank-DEMAND testset](https://datashare.ed.ac.uk/handle/10283/2791) and [DNS1 and DNS3](https://github.com/microsoft/DNS-Challenge)

Quality estimation (VQScore):

=> Training: [LibriSpeech clean-460 hours](https://www.openslr.org/12)

=> validation: [noisy speech of VoiceBank-DEMAND testset](https://datashare.ed.ac.uk/handle/10283/2791)

=> Evaluation: [Tencent and IUB](https://github.com/ConferencingSpeech/ConferencingSpeech2022/tree/main/Training/Dev%20datasets)

## Training
To Train our speech enhancement model (using only Clean Speech). Below is an example command.
```shell
python trainVQVAE.py \
-c config/SE_cbook_4096_1_128_lr_1m5_1m5_github.yaml \
--tag SE_cbook_4096_1_128_lr_1m5_1m5_github
```
To Train our speech quality estimator, VQScore. Below is an example command.
```shell
python trainVQVAE.py \
-c config/QE_cbook_size_2048_1_32_IN_input_encoder_z_Librispeech_clean_github.yaml \
--tag QE_cbook_size_2048_1_32_IN_input_encoder_z_Librispeech_clean_github
```

## Inference
Below is an example command for generating enhanced speech/ estimated quality scores from the model.
Where '-c' is the path of the config file, '-m' is the path of the pre-trained model, and '-i' is the path of the input wav file.

* Note: Because our training data is 16kHz clean speech, only 16kHz speech input is supported.
  
```shell
python inference.py \
-c ./config/SE_cbook_4096_1_128_lr_1m5_1m5_github.yaml \
-m ./exp/SE_cbook_4096_1_128_lr_1m5_1m5_github/checkpoint-dnsmos_ovr=2.761_AT.pkl \
-i ./noisy_p232_005.wav
```
```shell
python inference.py \
-c ./config/QE_cbook_size_2048_1_32_IN_input_encoder_z_Librispeech_clean_github.yaml \
-m ./exp/QE_cbook_size_2048_1_32_IN_input_encoder_z_Librispeech_clean_github/checkpoint-dnsmos_ovr_CC=0.835.pkl \
-i ./noisy_p232_005.wav
```
```shell
python inference_folder.py \
-c ./config/QE_cbook_size_2048_1_32_IN_input_encoder_z_Librispeech_clean_github.yaml \
-m ./exp/QE_cbook_size_2048_1_32_IN_input_encoder_z_Librispeech_clean_github/checkpoint-dnsmos_ovr_CC=0.835.pkl \
-i path to the folder you want to evaluate
```


## Pretrained Models
We provide the checkpoints of trained models in the corresponding ./exp/config_name folder.

* Note that the provided checkpoints are the models after we reorganize the code, so the results are slightly different from those shown in the paper.
* However, the overall trend should be similar.

## Adversarial noise
As shown in the following spectrogram, the applied adversarial noise doesn't have a fixed pattern as Gaussian noise. So it may be a good one to train a robust speech enhancement model. 
<center><img src="https://github.com/JasonSWFu/VQscore/blob/main/adv_wav.png" width="300"></center>

## Collaboration
I'm open to collaboration! If you find this Self-Supervised SE/QE topic interesting, please let me know (e-mail: szuweif@nvidia.com). 

### Citation
If you find the code useful in your research, please cite our ICLR paper :)
    
## References
* [vector-quantize](https://github.com/lucidrains/vector-quantize-pytorch) (for VQ-VAE)
* [DNSMOS](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS)
