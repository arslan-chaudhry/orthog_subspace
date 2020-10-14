This is the official implementation of the [Continual Learning in Low-rank Orthogonal Subspaces](https://arxiv.org/abs/2010.11635) in Tensorflow v1. The code is built on top of [AGEM repository](https://github.com/facebookresearch/agem).


```
@inproceedings{ChaudhryOrthogSubspaceCL,
    title={Continual Learning in Low-rank Orthogonal Subspaces},
    author={Chaudhry, Arslan and Khan, Naeemullah and Dokania, Puneet K and Torr, Philip HS},
    booktitle={NeurIPS},
    year={2020}
}
```

Download the miniImageNet dataset from [here](https://www.dropbox.com/s/yt3akdfchuafk25/miniImageNet_full.pickle?dl=0) and place it under the ```miniImageNet_Dataset/``` folder.

To replicate the results of the paper execute the following script:
```bash
$ ./replicate_results.sh
```

## License
This source code is released under The MIT License found in the LICENSE file in the root directory of this source tree.
