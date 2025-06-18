# TF-Codec: Latent-Domain Predictive Neural Speech Coding

Official implementation of the non-predictive version of the paper [Latent-Domain Predictive Neural Speech Coding](https://arxiv.org/abs/2207.08363).

<img src="assets/subjective_tfcodec.png" width="750">


## Prerequisites

- Python 3.10 and conda, get [Conda](https://www.anaconda.com/)
- CUDA 12.5 (other versions may also work. Make sure the CUDA version matches with pytorch.)
- pytorch 2.5 (We have tested that pytorch-2.5 works. Other versions may also work.)
- Environment
    ```
    conda create -n $YOUR_PY_ENV_NAME python=3.10
    conda activate $YOUR_PY_ENV_NAME
    pip install -r requirements.txt
    ```

## Pretrained models

Download [our pretrained models](https://1drv.ms/f/c/5fdaec1d5376d89a/EnLDJvj4S7JBscPWEKcEhcABZijlncQoX6K_Kdajt2IAQg?e=eVOMTi) and put them into ./checkpoints folder. Both the generator and discriminator weights are saved in the pretrained model ckpt.


## Training

Put your training and validation data (Multilingual_train.mdb and Multilingual_val.mdb in LMDB format) in ./training_data folder:

Stage-1 without adversarial training:
```bash
 python multiprocess_caller.py --nproc_per_node=4 --nnodes=1 --num_workers=2 --train_data_dir=training_data/Multilingual_train.mdb --val_data_dir=training_data/Multilingual_val.mdb --train_dir=job_tfcodec_stage1 --config=configs/tfcodec_config_train_stage1.yaml
```

Stage-2 finetuning from stage-1 checkpoints (./checkpoints/model_stage1.ckpt) with adversarial training:
```bash
python multiprocess_caller.py --nproc_per_node=4 --nnodes=1 --num_workers=2 --train_data_dir=training_data/Multilingual_train.mdb --val_data_dir=training_data/Multilingual_val.mdb --train_dir=job_tfcodec_stage2 --config=configs/tfcodec_config_train_stage2.yaml --checkpoint_path=checkpoints/model_stage1.ckpt
```


## Testing

Example to test pretrained models:
```bash
 python inf.py --audio_path=<input audio> --model_path=checkpoints/tfcodec_path/tfcodec_6k_514000.ckpt --config_path=configs/tfcodec_config_6k.yaml --output_path=<output audio>
 python inf.py --audio_path=<input audio> --model_path=checkpoints/tfcodec_path/tfcodec_1k_545000.ckpt --config_path=configs/tfcodec_config_1k.yaml --output_path=<output audio>
```
Only 16khz speech in .wav is supported currently. This version only provides encoding, quantization to token indices, and decoding modules. External huffman coding tools are needed to encode quantized token indices to a bitstream.


## Citation
If you find this work useful for your research, please cite:
```
@article{Jiang2023tfcodec,
  title={Latent-Domain Predictive Neural Speech Coding},
  author={Xue Jiang and Xiulian Peng and Huaying Xue and Yuan Zhang and Yan Lu},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={31},
  year={2023}
}
```

## Trademarks
This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.
