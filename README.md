# ProVision

<p align="center">
    <img src="pipeline.png" width="1000" style="margin-bottom: 0.2;"/>
<p>

If you like our project, please give us a star ⭐ on GitHub for the latest update.  </h2>




## What's ProVision?

ProVision is an extendable data generation engine that produces instruction data for large multimodal language models (MLMs). 

In particular, it synthesizes instruction data via data generators (Python programs) and scene graphs rather than proprietary models.
It also includes a scene graph generation pipeline consisting of various state-of-the-art models (e.g., object detection model). 
Thus, one can generate instruction data for any given image by first generating the scene graph and then applying data generators.

Provision supports the generation of both single-image and multi-image instruction data.
One can also extend the engine by adding new data generators.

## Usage

For augmented scene graph generation, please follow the instructions in [SCENE_GRAPH_GENERATION.md](SCENE_GRAPH_GENERATION.md).

For instruction generation, please refer to the demo notebook in the `notebook` folder.
It contains a step-by-step guide on 
1. how to generate single-image instruction data for images using ProVision's data generation engine
2. how to generate multi-image instruction data for images using ProVision's data generation engine

## ProVision-10M dataset

We release the ProVision-10M dataset, a 10M synthesized instruction data at this [link](https://huggingface.co/datasets/Salesforce/ProVision-10M)

## Disclaimers
**ProVision** and its associated resources are provided for research and educational purposes only. 
The authors and contributors make no warranties regarding the accuracy or reliability of the data and software. 
Users are responsible for ensuring their use complies with applicable laws and regulations. 
The project is not liable for any damages or losses resulting from the use of these resources.


## Contact

- Jieyu Zhang: jieyuz2@cs.washington.edu

## Citation

**BibTeX:**

```bibtex
@article{zhang2024provision,
  title={ProVision: Programmatically Scaling Vision-centric Instruction Data for Multimodal Language Models},
  author={Zhang, Jieyu and Xue, Le and Song, Linxin and Wang, Jun and Huang, Weikai and Shu, Manli and Yan, An and Ma, Zixian and Niebles, Juan Carlos and Xiong, Caiming and others},
  journal={arXiv preprint arXiv:2412.07012},
  year={2024}
}
```

