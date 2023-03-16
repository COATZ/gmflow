# Citation
This repositeory contains the optical flow model used in our article "OMNI-CONV: Generalization of the Omnidirectional Distortion-Aware Convolutions".

```
@article{Artizzu2023,
	title        = {{OMNI-CONV: Generalization of the Omnidirectional Distortion-Aware Convolutions}},
	author       = {Artizzu, Charles-Olivier and Allibert, Guillaume and Demonceaux, Cédric},
	year         = 2023,
	journal      = {Journal of Imaging},
	volume       = 9,
	number       = 2,
	article-number = 29,
	url          = {https://www.mdpi.com/2313-433X/9/2/29},
	pubmedid     = 36826948,
	issn         = {2313-433X},
	doi          = {10.3390/jimaging9020029}
}
```

# Installation
Create the python3 env and install packages:
```
python3 -m venv GMFLOW_ENV;
source GMFLOW_ENV/bin/activate;
pip3 install pytorch, tensorboard, opencv-python
```

# Spherical adaptation
Distortion-aware convolutions for the ENCODER are located in "backbone.py" file from line 85 to 86 and are activated by commenting the appropriate line:
```
        # self.conv1 = nn.Conv2d(3, feature_dims[0], kernel_size=7, stride=2, padding=3, bias=False)  # 1/2
        self.conv1 = DeformConv2d_sphe(3, feature_dims[0], kernel_size=7, stride=2, padding=3, bias=False)  # 1/2
```

Distortion-aware convolutions for the DECODER are located in "gmflow.py" file from line 47 to 81 and are activated by commenting the appropriate line:
```
        # self.upsampler = nn.Sequential(nn.Conv2d(2 + feature_channels, 256, 3, 1, 1),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0))
        self.upsampler = nn.Sequential(DeformConv2d_sphe(2 + feature_channels, 256, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0))

```


# Evaluation
Run the evaluation of the model on dataset "omni":
```
python3 main_v2.py --eval --val_dataset omni --output_path OUTPUT/ --resume ckpt/gmflow_things-e9887eda.pth
```

