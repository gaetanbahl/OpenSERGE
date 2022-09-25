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
