<div align="center">
  
# Neural Panoramic Representation for Spatially and Temporally Consistent 360° Video Editing

</div>

<p align="center">
  <a href="https://scholar.google.com/citations?user=mIkY9yIAAAAJ&hl=en">Simin Kou</a> • 
  <a href="https://fanglue.github.io/">Fang-Lue Zhang</a> • 
  <a href="https://users.cs.cf.ac.uk/Yukun.Lai/">Yu-Kun Lai</a> • 
   <a href="https://people.wgtn.ac.nz/neil.dodgson">Neil A. Dodgson</a><br>
  <br>
  <a href="https://ieeexplore.ieee.org/abstract/document/10765439">
    <img src="https://img.shields.io/badge/IEEE%20ISMAR-2024-blue">
  </a>
</p>

<p align="left">
  <img src="assets/teaser.png" width="900"><br>
  <small>The framework of our proposed <strong>N</strong>eural <strong>P</strong>anoramic <strong>R</strong>epresentation (<strong>NPR</strong>).</small> <be> <sub> Our model represents 360° videos using MLPs, allowing for easy video editing in the true spherical space. Given the captured 360° video, its segmentation masks, and the designed 4D spatiotemporal coordinates as inputs, our model predicts implicit spherical positions for generating spherical content layers, providing each layer's appearance for reconstruction. We incorporate bi-directional mapping by introducing an additional pair of backward mapping MLPs to model the global motion of individual dynamic scenes, facilitating flexible 360° video editing.</sub>
</p>

### 📜 Cite This Paper
```bibtex
@inproceedings{kou2024neural,
  title={Neural Panoramic Representation for Spatially and Temporally Consistent 360° Video Editing},
  author={Kou, Simin and Zhang, Fang-Lue and Lai, Yu-Kun and Dodgson, Neil A},
  booktitle={2024 IEEE International Symposium on Mixed and Augmented Reality (ISMAR)},
  pages={200--209},
  year={2024},
  organization={IEEE}
}
```

## 💰 Funding
This research was supported by Marsden Fund Council managed by the Royal Society of New Zealand under Grant MFP-20-VUW-180 and the Royal Society (UK) under Grant No. IES\R1\180126.


## ⚙️ Installation
**Clone the repository**  
```sh
git clone https://github.com/SiminKoux/neural-panoramic-representation.git
cd neural-panoramic-representation
```

**Environment setup**  
```sh
# Create a conda environment
conda create -n npr python=3.9

# Activate env
conda activate npr

# Install dependencies
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

## 📂 Dataset
We have included the entire dataset in this repository as the ``/data`` folder.<br>
You can also download the dataset from Hugging Face if you only need to use our dataset:
```sh
git clone git@hf.co:datasets/SiminKou/360NPR.git
```
## 🚀 Training
To start the training process, run the following command:
```sh
bash run.sh
```
If you want to train a model for a different video (e.g., replacing the example ``Walking_Boy``), update the ``type`` and ``name`` in the ``/configs/data.yaml`` file with the desired values, such as ``Walking_Girl``.

**Notes:**
The warmstart checkpoints provided in this repository are specifically for the ``Walking_Boy`` video. 
- The files ``warmstart_uvw_mapping_f.pth`` and ``warmstart_uvw_mapping_b.pth`` can be used for any video with the same resolution without requiring additional training.
- However, ``warmstart_alpha_pred.pth`` requires warmstart training for each new video beyond the provided ``Walking_Boy``.
To warmstart alpha mapping for a new video, modify the ``configs/model.yaml`` file by setting the following parameters to ``False``:
```yaml
load_checkpoint: False
warmstart_mapping1: False
warmstart_mapping1: False
main_train: False
```
No changes are needed for other parameters in this file. After completing the warmstart training for alpha mapping, enable the above four parameters and proceed with the main training process.
