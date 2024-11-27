# ADMM-inspired Network for Nonnegative Matrix Factorization

This project provides the code for the paper titled **"ADMM-inspired Network for Nonnegative Matrix Factorization"**.

## Requirements

To run this project, please ensure you have the following packages installed (`Python version: 3.9.19`):

- `numpy==1.24.1`
- `pandas==2.2.2`
- `scikit_learn==1.4.2`
- `scipy==1.13.0`
- `torch==2.2.2`

## Running Environment in Paper

- **Operating System:** Windows 11
- **CPU:** R7 7840HS
- **RAM:** 64G
- **GPU:** NVIDIA GeForce RTX 4050

## Citations

This project builds upon the following code repositories:

1. [NMFLibrary: Non-negative Matrix Factorization Library](https://github.com/hiroyuki-kasai/NMFLibrary)
2. [Differentiable Linearized ADMM](https://github.com/zs-zhong/D-LADMM)



## Example

The MNIST dataset is stratified sampled into 10 equal parts. The first slice is used to train LNMF, and the other slices are directly used as input to obtain the corresponding decomposition sub-matrices W and H. Then, H is used as input (the low-dimensional matrix of the original data) for classification experiments, and the corresponding classification accuracy (AC) is output. The following is the optimal average of 20 epochs. It can be seen that LNMF can quickly obtain the decomposition sub-matrices of other data slices under the same distribution, avoiding the computational cost of re-computation.

### Data Adaptability

| Test Data | Accuracy |
|-----------|----------|
| data_slice2 | 0.82429 |
| data_slice3 | 0.80667 |
| data_slice4 | 0.80476 |
| data_slice5 | 0.79714 |
| data_slice6 | 0.80571 |
| data_slice7 | 0.82048 |
| data_slice8 | 0.80429 |
| data_slice9 | 0.79952 |
| data_slice10 | 0.81571 |
| Average | 0.80873 |
