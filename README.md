# Efficirnt-AoADet-RADDet

## abstract
Millimeter-wave radar is essential for robust perception in autonomous systems, as its performance remains independent of visual conditions. However, conventional multi-stage radar object detection algorithms often face challenges in real-time deployment due to computational inefficiency and limited task adaptability. To solve this engineering application challenge, we propose an efficient and physics-guided framework that integrates domain knowledge with data-driven learning to address these efficiency gaps. Unlike conventional methods, our proposed method operates directly on the compact complex-valued range-Doppler radar spectrum. To effectively extract information from this representation, our framework incorporates three jointly optimized modules. First, a complex-valued virtual channel decoupling module utilizes waveform structure priors to decouple virtual channels and eliminate Doppler aliasing. Second, a learnable complex-valued Angle of Arrival (AoA) estimation module, initialized from the Fast Fourier Transform (FFT) basis, replaces static angle-FFT to provide task-specific angular refinement at significantly lower cost. Finally, an Inter-Dimensional-aware attention mechanism exploits the anisotropic nature of radar data to enhance feature expressiveness. This endto-end, multi-task joint optimization mechanism allows the entire system to learn task-aligned representations, boosting both robustness and efficiency. Extensive experiments demonstrate that our method significantly reduces overhead (up to 48.9% fewer parameters, 52.8% reduction in floating-point operations, and 7.6-fold faster inference) while achieving an 18.6% average improvement in detection accuracy over advanced multistage approaches. These results highlight a scalable and high-precision solution for real-time, intelligent radar perception on resource-constrained platforms.

## Project Overview
This project is the Efficient-AoADet validated on the RADDet dataset, with the code implemented in PyTorch (version 2.0.1 + CUDA 118). The training script is named train_cart_w0_RADSuperivised.py, while all quantization and visualization evaluation codes are located in the "evaluates" directory.

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{pang2026physics,
  title={Physics-guided efficient automotive radar object detection framework via multi-task joint optimization},
  author={Pang, Siqi and Guo, Kaitai and Zheng, Yang and Liang, Jimin},
  journal={Engineering Applications of Artificial Intelligence},
  volume={176},
  pages={114724},
  year={2026},
  publisher={Elsevier}
}
