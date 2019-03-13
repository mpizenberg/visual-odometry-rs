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
# format: depth_timestamp depth_file_path rgb_timestamp rgb_file_path
1305031102.160407 depth/1305031102.160407.png 1305031102.175304 rgb/1305031102.175304.png
1305031102.226738 depth/1305031102.226738.png 1305031102.211214 rgb/1305031102.211214.png
1305031102.262886 depth/1305031102.262886.png 1305031102.275326 rgb/1305031102.275326.png
...
```

### tum-read-trajectory

## Candidates

### coarse-to-fine

### dso

## Optimization

### regression-1d

### rosenbrock

### affine-2d
