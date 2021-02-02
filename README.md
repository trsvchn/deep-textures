# Deep Textures

<a href="https://colab.research.google.com/github/trsvchn/deep-textures/blob/main/example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Unofficial Reimplementation of "Texture Synthesis Using Convolutional Neural Networks" in PyTorch
(Gatys et al. [1])

## Dependencies

`python=3.8.5` `pytorch=1.7.1` `torchvision=0.8.2` `pillow=8.1.0`

## Notes

This implementation uses pretrained `vgg19` model from the `torchvision` model zoo, thus model weights are
not scaled as described in the original paper.

## References

1. Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). Texture synthesis using convolutional neural networks.
   arXiv preprint [arXiv:1505.07376](http://arxiv.org/abs/1505.07376)
2. [Official Caffe implementation](https://github.com/leongatys/DeepTextures)

## Disclaimer

This software is published for academic and non-commercial use only.
