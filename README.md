# StellarGAN: Classifying Stellar Spectra with Generative Networks in SDSS and APOGEE Sky Surveys
<div align="center">
  <img src="Figures/stellarGAN.png" width="900px" />
    Figure 1
</div>
Figure 1. The structure schematic of stellarGAN in the pre-training phase. The noise vector is fed into G to generate spectra, which are labeled as 0 to indicate that they are not from real data. Then the generated spectra and real spectra (labeled as 1) are sent to train the D.

### Dependencies

Our implementation uses external libraries such as NumPy and PyTorch. You can resolve the dependencies with the following command.
```
pip install numpy
pip install -r requirements.txt

Note that this command may dump errors during installing pycocotools, but the errors can be ignored.

### Dataset

#### SDSS and APOGEE
SDSS and APOGEE dataset can be downloaded [here](https://www.sdss.org/).

```
stellarGAN
 |─ data
 │   └─ SDSS
 |       |─ A-type
 |       |─ F-type
 |       |─ G-type
 |       |─ K-type
 |       |─ O-type
 |       |─ B-type
 |       |─ M-type
  │   └─ APOFEE
 |       |─ A-type
 |       |─ F-type
 |       |─ G-type
 |       |─ K-type
 |       |─ M-type

 



```

