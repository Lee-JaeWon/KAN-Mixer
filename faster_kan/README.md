# FasterKAN = FastKAN + RSWAF bases functions and benchmarking with other KANs
  
This repository contains a very fast implementation of Kolmogorov-Arnold Network (KAN). As of 5/17/2024 is the fastest implementation according to my knowledge and benchmarks. As of 5/25/2024 2nd/3rd fastest with ChebyKAN faster and rbf-kan faster in forward based on [#issue 6](https://github.com/AthanasiosDelis/faster-kan/issues/7). I will test with my newest version towards the end of the 1st week of June.

In the latest edition user can choose if the grid is a learnable parameter. (Still experimental functionality as of 5/17/2024)

It uses approximations of the B-spline using the Switch Activation Function, inspired from the work of [Bo Deng](https://digitalcommons.unl.edu/mathfacpub/68/),
modified for having Reflectional symmetry. This kind of Activation Function can approximate 3rd order B-Splines used in the original [pykan](https://github.com/KindXiaoming/pykan).


The original implementation of KAN is [pykan](https://github.com/KindXiaoming/pykan).

The original implementation of Fast-KAN is [fast-kan](https://github.com/ZiyaoLi/fast-kan). FasterKAN uses as its bases the code from Fast-KAN.

The forward time of FaskKAN is 3.33x faster than [efficient KAN](https://github.com/Blealtan/efficient-kan), and the implementation is a LOT easier for FaskKAN vs efficient KAN.

The forward time of FaskerKAN is 1.5x faster than [fast-kan](https://github.com/ZiyaoLi/fast-kan).



[fast-kan](https://github.com/ZiyaoLi/fast-kan) used Gaussian Radial Basis Functions to approximate the B-spline basis, which is the bottleneck of KAN and efficient KAN:

$$b_{i}(u)=\exp\left(-\left(\frac{u-u_i}{h}\right)^2\right)$$

## FasterKAN
1. Here in [faster-kan](https://github.com/AthanasiosDelis/faster-kan), the idea is to experiment with other bases, exponent values, h values and the exclusion of the bases SiLU Funciton.

    For the momement Reflectional SWitch Activation Function (RSWAF) functions seem the most promising to use:

    $$b_i(u) = 1 - \left(\tanh\left(\frac{u - u_i}{h}\right)\right)^2$$


    The rationale of doing so is that these RSWAF functions can approximate the B-spline basis (up to a linear transformation), they are easy to calculate while having uniform grids.
	
    In the latest edition user can choose if the inverse of the denominator (1/h) is a learnable parameter. (Still experimental functionality as of 5/17/2024)
  
    Results of approximation of a 3-rd order spline for a [28*28,256,10] efficient-KAN are shown in the figure below (code in [notebook](draw_spline_basis.ipynb)). 

    ![RSWAF well approximates 3-order B-spline basis.](https://github.com/AthanasiosDelis/faster-kan/blob/master/img/compare_basis.png)


2. Used LayerNorm to scale inputs to the range of spline grids, so there is no need to adjust the grids.

3. From [fast-kan](https://github.com/ZiyaoLi/fast-kan):

    FastKAN is 3.33x compared with efficient_kan in forward speed, based on [ZiyaoLi](https://github.com/ZiyaoLi)

    The benchmarking I tested inspired from [KAN-benchmarking](https://github.com/Jerry-Master/KAN-benchmarking),
    indicates that FastKAN may be even faster than originally though and FasterKAN is the fastest of all for the time being.

    Experiments were executed on a NVIDIA GeForce RTX3050 Ti 4G and an AMD Ryzen 7 6800H, and the network has dimensions [28x28,256,10],
    except from the MLP that has a hidden layer 256*5 to match the num params of FasterKAN.

    It seems that the various KAN implementations yield different num params for the same hidden layer and B-spline order (or its equivilent approximation parameters in other implementions) :

    |                 | forward	 | backward	 | forward	 | backward	 | num params	 | num trainable params	 |
    |-----------------|----------|-----------|-----------|-----------|-----------|-----------|
    | fasterkan-gpu     | 0.61 ms	 | 1.41 ms	 | 0.06 GB	 | 0.06 GB	 | 3254336	 | 3254304	 |
    | fastkanorg-gpu     | 0.84 ms	 | 1.73 ms	 | 0.05 GB	 | 0.06 GB	 | 3457866	 | 3457834	 |
    | mlp-gpu     | 0.29 ms	 | 0.79 ms	 | 0.04 GB	 | 0.05 GB	 | 3256330	 | 3256330	 |
    | effkan-gpu     | 2.80 ms	 | 2.74 ms	 | 0.04 GB	 | 0.05 GB	 | 2032640	 | 2032640	 |
    | kalnet-gpu     | 1.57 ms	 | 2.13 ms	 | 0.10 GB	 | 0.10 GB	 | 1016852	 | 1016852	 |

    FasterKAN is ~1.5 faster than FastKAN and ~2 slower from MLP in forward speed

    FasterKAN is 1.22 faster than FastKAN and 1.75 slower from MLP in backward speed

    FasterKAN is 0.83 smaller than FastKAN and 1.5 bigger from MLP in forbward memory

    FasterKAN is equal with FastKAN and 1.2 bigger from MLP in backward memory

4. Train on mnist with a FasterKAN[28*28,256,10] with 815144 num trainable params for 15 epochs, 97% accuracy is achieved:

    100%|█| 938/938 [00:13<00:00, 67.87it/s, accuracy=1, loss=0.013, lr=0.00010

    Epoch 15, Val Loss: 0.07934391943353768, Val Accuracy: 0.9772093949044586

5. Train on mnist with a FasterKANvolver[ 64, 10] with 1,213,888 num trainable params for 15 epochs,  99.582% accuracy is achieved:
 
    100%|█| 938/938 [00:17<00:00, 52.80it/s, accuracy=1, loss=4.31e-5, lr=1.68e

    Epoch 00015: reducing learning rate of group 0 to 1.0078e-05.

    Epoch 15, Val Loss: 0.020724332156509737, Val Accuracy: 0.9958200636942676

    Current Learning Rate: 1.0077695999999996e-05
 
6. Train on mnist with a FasterKANvolver[ 64, 32, 16, 10] with 1,230,624 num trainable params for 15 epochs, 99.5023% accuracy is achieved.

    EnhancedFeatureExtractor and Training Pipiline is by no means optimized, FasterKANvolver works only as a proof of concept that KAN and MLPs can coexist seamlessly with good results.

    I remind the [SOTA](https://paperswithcode.com/paper/a-branching-and-merging-convolutional-network) is 99.87% 

7. The most important thing is to test if [pykan](https://github.com/KindXiaoming/pykan) indeed has the continuous learning capabilities that it promises and if [fast-kan](https://github.com/ZiyaoLi/fast-kan) and [faster-kan](https://github.com/AthanasiosDelis/faster-kan) inherit these capabilities as well as the ability for pruning and symbolic regression.

## Contact

For any inquiries or support, contact: athanasiosdelis@mail.ntua.gr or adelis2@jhu.edu

## Citation

Copyright 2024 Athanasios, Delis. Licensed under the Apache License, Version 2.0 (the "License");

```bibtex
@misc{Athanasios2024,
  author = {Athanasios Delis},
  title = {FasterKAN},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AthanasiosDelis/faster-kan/}}
}
```

## Contributions

Contributions are welcome. Please raise issues as necessary. All issues, as they come up, will be definitely solved to the best of my abilities. Till then if you send a merge request, describe the problem, the fix and why it works.

Thanks to [Da1sypetals](https://github.com/Da1sypetals) for his cuda contribution.

## References

- [0] Ziming Liu et al., "KAN: Kolmogorov-Arnold Networks", 2024, arXiv. https://arxiv.org/abs/2404.19756
- [1] https://github.com/KindXiaoming/pykan
- [2] https://github.com/Blealtan/efficient-kan
- [3] https://github.com/ZiyaoLi/fast-kan/tree/master
- [4] https://github.com/1ssb/torchkan/tree/main
- [5] https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#Tanh
