# OpenSERGE

Open source re-implementation of [**Single Shot End-to-end (Road) Graph Extraction** (SERGE) presented at **CVPR EARTHVISION 2022**.](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/html/Bahl_Single-Shot_End-to-End_Road_Graph_Extraction_CVPRW_2022_paper.html)

**Work in progress, star to stay tuned! :)**

## Abstract:

Automatic road graph extraction from aerial and satellite images is a long-standing challenge. Existing algorithms are either based on pixel-level segmentation followed by vectorization, or on iterative graph construction using next move prediction. Both of these strategies suffer from severe drawbacks, in particular high computing resources and incomplete outputs. By contrast, we propose a method that directly infers the final road graph in a single pass. The key idea consists in combining a Fully Convolutional Network in charge of locating points of interest such as intersections, dead ends and turns, and a Graph Neural Network which predicts links between these points. Such a strategy is more efficient than iterative methods and allows us to streamline the training process by removing the need for generation of starting locations while keeping the training end-to-end. We evaluate our method against existing works on the popular RoadTracer dataset and achieve competitive results. We also benchmark the speed of our method and show that it outperforms existing approaches. Our method opens the possibility of in-flight processing on embedded devices for applications such as real-time road network monitoring and alerts for disaster response.


If you found this work interesting and used it for your research, consider citing:

```
@InProceedings{Bahl_2022_CVPR,
    author    = {Bahl, Gaetan and Bahri, Mehdi and Lafarge, Florent},
    title     = {Single-Shot End-to-End Road Graph Extraction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {1403-1412}
}


This repository is a clean-room, open-source re-implementation of the approach described in:

> *Single-Shot End-to-End Road Graph Extraction*, CVPRW 2022.  
> G. Bahl, M. Bahri, F. Lafarge.

**High-level idea**: a fully-convolutional head detects road **points of interest** (junctions/turns/dead-ends) and regresses a 2D **offset** per cell; a lightweight GNN then predicts **edges** between detected points to produce a vector road graph in one pass.

## What’s here
- Minimal, framework-only PyTorch code (no third-party GNN libs required).
- A tiny, readable GNN with **EdgeConv** and an MLP edge scorer.
- A junction/offset detection head built on a standard CNN backbone.
- Training & eval scaffolds, metrics hooks, and dataset stubs (RoadTracer-style).

> ⚠️ This is a **reference scaffold**: you’ll need to connect your dataset and metrics (APLS, J-F1, P-F1). The code includes clear TODOs.

## Quick start (conda)
```bash
conda create -n openserge python=3.10 -y
conda activate openserge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # adjust CUDA/CPU
pip install numpy opencv-python shapely networkx tqdm
```

## Train (example)
```bash
python -m openserge.train   --data_root /path/to/RoadTracer   --epochs 50   --batch_size 4   --lr 1e-3   --backbone resnet50   --img_size 512   --junction_thresh 0.5
```

## Inference (tile an 8192² image)
```bash
python -m openserge.infer   --weights /path/to/ckpt.pt   --image /path/to/8192.png   --img_size 512   --stride 448   --junction_thresh 0.5   --k 4   --save_graph out.graph.json
```

## Notes
- Metrics and exact training recipes vary per dataset; see inline comments for recommended settings from the paper.
- For reproducible speed, consider export to TorchScript, quantization-aware training, and smaller backbones.

## License
MIT. See `LICENSE`.
