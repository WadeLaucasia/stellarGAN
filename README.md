# StellarGAN: Classifying Stellar Spectra with Generative Networks in SDSS and APOGEE Sky Surveys
<div align="center">
  <img src="Figures/stellarGAN.png" width="900px" />
</div>
Figure 1. The structure schematic of stellarGAN in the pre-training phase. The noise vector is fed into G to generate spectra, which are labeled as 0 to indicate that they are not from real data. Then the generated spectra and real spectra (labeled as 1) are sent to train the D.

### Dependencies

Our implementation uses external libraries such as NumPy and PyTorch. You can resolve the dependencies with the following command.
```
pip install numpy
pip install -r requirements.txt
```
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
## Training
```
Step 1: Pre-training the Discriminator (D)

The pre-training process for the Discriminator can be initiated using the following Python command, which runs the script train_pre.py:

python train_pre.py
```
Within this script, the Discriminator model ('D') is trained for a binary classification task. Once the training is complete, the pre-trained model parameters are saved for future use.

Step 2: Initialize and Re-train the Discriminator (D)
After the pre-training is complete, you can remove any instances of the Generator model ('G') if they exist. Subsequently, initialize a new Discriminator model using the pre-trained parameters. To adapt the model for a multi-class classification task, modify its architecture accordingly.

The training for this adapted Discriminator can be executed with the following command:
```
python train_sec_D.py
```
