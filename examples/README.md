# Examples

Examples showing usage of this library are located in this directory.
They are regrouped by common prefix since we cannot have sub directories.

- `dataset_...`: examples showing how to load data from supported datasets.
- `candidates_...`: examples showing how to pick sparse candidate points for tracking.
- `optim_...`: examples showing how the `OptimizerState` trait
  can be implemented on problems with different complexities.

To run one example, move to the root of this repository
(where the `Cargo.toml` file lives) and invoke the command:

```sh
cargo run --release --example example_name [example_arguments]
```

## Dataset

### tum-read-associations

This example shows how to read the associations file
of a dataset following the TUM RGB-D format.
Such association file contains data similar to:

```txt
# depth_timestamp depth_file_path rgb_timestamp rgb_file_path
1305031102.160407 depth/1305031102.160407.png 1305031102.175304 rgb/1305031102.175304.png
1305031102.226738 depth/1305031102.226738.png 1305031102.211214 rgb/1305031102.211214.png
1305031102.262886 depth/1305031102.262886.png 1305031102.275326 rgb/1305031102.275326.png
...
```

### tum-read-trajectory

This example shows how to read the trajectory file
of a dataset following the TUM RGB-D format.
Such trajectory file contains data similar to:

```txt
# ground truth trajectory
# timestamp tx ty tz qx qy qz qw
1305031098.6659 1.3563 0.6305 1.6380 0.6132 0.5962 -0.3311 -0.3986
1305031098.6758 1.3543 0.6306 1.6360 0.6129 0.5966 -0.3316 -0.3980
...
```

## Candidates

Candidate points are a subset of an image pixels
suitable for tracking, to enable direct visual odometry.
Traditionally, "indirect" approches try to track key points such as SIFT or ORB.
In a direct approch, we do not track key points, but we still need
a sparse set of points to track for beeing able to minimize our energy function.
The selected points should usually satisfy two conditions:

1. Points shall be well distributed in the image.
2. The density of candidate points grows with the gradient magnitude.

### coarse-to-fine

In this approach, candidates are selected by a coarse to fine mechanism.
We build a pyramid of gradient magnitude images,
where each level is half the resolution of the previous one.
At the lowest resolution, we consider all points to be candidates.
For each double resolution,
for each 2x2 block corresponding to a candidate at the sub resolution,
we select 1 or 2 of the pixels as candidates depending on a criteria.

See for example the following image showing candidate points
(in red) at each level of a 4-levels image pyramid.
They contain respectively 609, 948, 1546 and 2636 candidate points.

![candidates coarse to fine][candidates-coarse-to-fine]

[candidates-coarse-to-fine]: https://mpizenberg.github.io/resources/vors/candidates-coarse-to-fine.png

### dso

## Optimization

### regression-1d

### rosenbrock

### affine-2d
