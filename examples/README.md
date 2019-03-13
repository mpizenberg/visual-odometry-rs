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

### tum-read-trajectory

## Candidates

### coarse-to-fine

### dso

## Optimization

### regression-1d

### rosenbrock

### affine-2d
