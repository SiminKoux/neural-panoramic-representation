<div align="center">
  
# Neural Panoramic Representation for Spatially and Temporally Consistent 360° Video Editing

</div>

<p align="center">
  <a href="https://scholar.google.com/citations?user=mIkY9yIAAAAJ&hl=en">Simin Kou</a> • 
  <a href="https://fanglue.github.io/">Fang-Lue Zhang</a> • 
  <a href="https://users.cs.cf.ac.uk/Yukun.Lai/">Yu-Kun Lai</a> • 
   <a href="https://people.wgtn.ac.nz/neil.dodgson">Neil A. Dodgson</a>
</p>

<p align="center">
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
