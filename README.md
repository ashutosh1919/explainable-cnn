# Explainable CNNs
[![Torch Version](https://img.shields.io/badge/torch-1.10.0-61DAFB.svg?style=flat-square)](#torch) [![Torchvision Version](https://img.shields.io/badge/torchvision-0.2.2-yellow.svg?style=flat-square)](#torchvision) <!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section --> [![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors-) <!-- ALL-CONTRIBUTORS-BADGE:END --> [![Price](https://img.shields.io/badge/price-free-ff69b4?style=flat-square)](#price) [![Maintained](https://img.shields.io/badge/maintained-yes-green.svg?style=flat-square)](#maintained)

Its a common notion that a Deep Learning model is considered as a black box. Working towards this problem, this project focusses on making the internal working of the Neural layers more transparent. In order to do so, Explainable CNNs is a plug n play component that visualizes the layers based on on their gradients and builds different representations including Saliency Map, GuidedBackPropagation, GradCam and GuidedgradCam. 

### Architechture

<p align="center">
<img src = "https://github.com/ashutosh1919/explainable-cnn/blob/main/data/architecture.png"></img>
</p>

### Usage

Install the package 

```
pip install explainable-cnn
```

To create visualizations, create an instance of `CNNExplainer`

```
from explainable_cnn import CNNExplainer

x_cnn = CNNExplainer(...)
```

The following method calls returns different visualizations 

```
saliency_map = x_cnn.get_saliency_map(...)

grad_cam = x_cnn.get_grad_cam(...)

guided_grad_cam = x_cnn.get_guided_grad_cam(...)
```

<p>To see full list of arguments and their usage for all methods, please refer to <a href="https://github.com/ashutosh1919/explainable-cnn/blob/main/src/explainable_cnn/explainers/cnn_explainer.py">this file</a></p>
<p>You may want to look at example usage in the <a href="https://github.com/ashutosh1919/explainable-cnn/blob/main/examples/explainable_cnn_usage.ipynb">example notebook</a>.</p>

### Output
<p>Below is a comparison of the visualization generated between GradCam and GuidedGradCam </p>

<p align="center"> 
    <img src="https://github.com/ashutosh1919/explainable-cnn/blob/main/data/outputs/explainable-cnn-output.png" align="center" height="1000px"></img>
</p>

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/ashutosh1919"><img src="https://avatars.githubusercontent.com/u/20843596?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Ashutosh Hathidara</b></sub></a><br /><a href="https://github.com/ashutosh1919/explainable-cnn/commits?author=ashutosh1919" title="Code">💻</a> <a href="#design-ashutosh1919" title="Design">🎨</a> <a href="#research-ashutosh1919" title="Research">🔬</a></td>
    <td align="center"><a href="https://github.com/L-Pandey"><img src="https://avatars.githubusercontent.com/u/90662028?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Lalit Pandey</b></sub></a><br /><a href="#research-L-Pandey" title="Research">🔬</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

### References

- [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/pdf/1610.02391.pdf)
- [Grad CAM demonstrations in PyTorch](https://github.com/kazuto1011/grad-cam-pytorch)
