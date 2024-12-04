# SiamABC: Improving Accuracy and Generalization for Efficient Visual Tracking
<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2411.00683-red)](https://arxiv.org/pdf/2411.18855)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://wvuvl.github.io/SiamABC_page/)

[Ram Zaveri](https://ramzaveri.com/),
[Shivang Patel](https://www.shivangapatel.com/),
[Yu Gu](https://directory.statler.wvu.edu/faculty-staff-directory/yu-gu),
[Gianfranco Doretto](https://vision.csee.wvu.edu/people/gianfranco-doretto/)

WACV 2025
</div>

This repository is the official implementation of [SiamABC](https://arxiv.org/pdf/2411.18855), an single object tracker designed for efficiently tracking under adverse visibility conditions.


![](docs/static/images/approach.png)
The Feature Extraction Block uses a readily available backbone to process the frames. The RelationAware Block exploits representational relations among the dual-template and dual-search-region through our losses, where dual-template and dual-search-region representations are obtained via our learnable FMF layer. The Heads Block learns lightweight convolution layers to infer the bounding box and the classification score through standard tracking losses. During inference, the tracker adapts to every instance through our Dynamic Test-Time Adaptation framework.

## OOD Comparison
<p align="center">
  <img src="docs/static/images/fig_quant.png" alt="method" style="max-width: 50%;">
</p>

Comparison of our trackers with others on the AVisT dataset on a CPU. We show the success score (AUC) (vertical axis), speed (horizontal axis), and relative number of FLOPs (circles) of the trackers. Our trackers outperform other efficient trackers in terms of both speed and accuracy.

## Dynamic Test-Time Adaptation
<p align="center">
  <img src="docs/static/images/table_4.png" alt="method" style="max-width: 70%;">
</p>

## AVisT, NFS30, UAV123, TrackingNet, GOT-10k, and LaSOT benchmarks
<p align="center">
  <img src="docs/static/images/table_2.png" alt="method" style="max-width: 70%;">
</p>


## Environment setup
The training code is tested on Linux systems.
```shell
conda create -n SiamABC python=3.7
conda activate SiamABC
pip install -r requirements.txt
```

## Single Video Evaluation

The SiamABC model is available in the `assets/model.pt`.  Run the following code:
```shell
python realtime_test.py --initial_bbox=[416, 414, 61, 97] --video_path=assets/penguin_in_fog.mp4 --output_path=outputs/penguin_in_fog.mp4
```

## Citation

```bibtex
@inproceedings{zaveri2025siamabc,
    title={Improving Accuracy and Generalization for Efficient Visual Tracking},
    author={Zaveri, Ram and Patel, Shivang and Gu, Yu and Doretto, Gianfranco},
    booktitle={Winter Conference on Applications of Computer Vision},
    year={2025},
    organization={IEEE/CVF}
}
```

## Additional Links
* [Vision and Learning Group ](https://vision.csee.wvu.edu/)

## Acknowledgement
* We thank  [FEAR](https://github.com/PinataFarms/FEARTracker) for the base code and  [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) for making the evaluation kit for object trackers.