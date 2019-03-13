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

### coarse-to-fine

### dso

## Optimization

### regression-1d

### rosenbrock

### affine-2d
