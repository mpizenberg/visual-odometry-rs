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

You can run this candidates selection algorithm on an image of your choice as follows.
This will create multiple png images named `candidates_0.png`, `candidates_1.png` ...
in the directory where the image lives.

```sh
cargo run --release --example candidates_coarse-to-fine /path/to/image
```

[candidates-coarse-to-fine]: https://mpizenberg.github.io/resources/vors/candidates-coarse-to-fine.png

### dso

The candidates selection in [DSO][dso] is quite different, and requires far more parameters.
At its heart, it consists of picking the highest gradient magnitude point per block.
A block being a rectangular group of pixels.
In practice there are many more complications.
A pixel is only selected if its gradient magnitude is higher than a threshold,
computed as the sum of a global constant, and a the median gradient magnitude of the current "region".
Regions being like blocks but bigger.
There are other details, like multiple levels, with lower thresholds,
and possible random sub-selection to achieve an objective amount of points.

You can run this candidates selection algorithm on an image of your choice as follows.
This will create an image named `candidates.png` in the directory where the image lives.

```sh
cargo run --release --example candidates_dso /path/to/image
```

[dso]: https://github.com/JakobEngel/dso

## Optimization

In order to solve the non-linear problem of camera tracking by some energy minimization,
there was a need to implement non-linear iterative solvers.
In particular, I implemented a [Levenberg-Marquardt][levenberg] least square optimization.
In the library code, the two modules inside `core::track` are an implementation of such algorithm.
They minimize a reprojection error, in an [inverse compositional approach][baker],
and a parameterization of the [rigid body motion][rigid-transformation]
in the Lie algebra of twists [se3][lie].

[levenberg]: https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
[baker]: http://www.ncorr.com/download/publications/bakerunify.pdf
[rigid-transformation]: https://en.wikipedia.org/wiki/Rigid_transformation
[lie]: http://ethaneade.com/lie.pdf

After a lot of API trial and error,
I ended with a code structure suitable for any iterative optimization algorithm.
This "structure" has been formalized by the trait `OptimizerState` in the module `math::optimizer`.
It basically goes as follows (see documentation for details).

```rust
pub trait OptimizerState<Observations, EvalState, Model, Error> {
    fn init(obs: &Observations, model: Model) -> Self;
    fn step(&self) -> Result<Model, Error>;
    fn eval(&self, obs: &Observations, new_model: Model) -> EvalState;
    fn stop_criterion(self, nb_iter: usize, eval_state: EvalState) -> (Self, Continue);
    fn iterative_solve(obs: &Observations, initial_model: Model) -> Result<(Self, usize), Error> {
        ...
    }
}
```

It means that if you implement `init`, `step`, `eval` and `stop_criterion` for a struct
of your custom type `MyOptimizer`, you will be able to call
`MyOptimizer::iterative_solve(obs, model)` to get the solution.
The implementation of `iterative_solve` is quite straightforward so don't hesitate to have a look at it.

Details about the generic types and the four functions to implement are in the documentation.
Simpler use cases than the camera tracking one are present in the following examples.

### regression-1d

![Regression of exponential data][optim_regression-1d]

[optim_regression-1d]: https://mpizenberg.github.io/resources/vors/optim_regression-1d.svg

In this example, we implement the `OptimizerState` trait
to find the correct parameter `a` for modelling exponentially decreasing noisy data.
It is implemented using a [Levenberg-Marquardt][levenberg] approach,
but simpler approaches would have also worked.
Details of the computation of the Jacobian and Hessian approximation are
provided at the beginning of the example file.

You can run the example as follows:

```sh
cargo run --releas --example optim_regression-1d
```

### rosenbrock

![Rosenbrock function][rosenbrock-png]

_3D visualization of the Rosenbrock function,
By Morn the Gorn - Own work, Public Domain_

The [Rosenbrock function][rosenbrock] is very often used to compare optimization algorithms.
It is defined by: `f: (a, b, x, y) -> (a-x)^2 + b*(y-x^2)^2`.

Given `a` and `b` fixed as constants,
the global minimum of the rosenbrock function is obtained for `(x, y) = (a, a^2)`
where its value is 0.
Again, the implementation in the example is using the Levenberg-Marquardt algorithm.
In the example code, we define `(a, b) = (1, 100)` and so the minimum is obtained for
`(x, y) = (1, 1)`.
You can run the example as follows:

```sh
cargo run --release --example optim_rosenbrock
```

[rosenbrock-png]: https://mpizenberg.github.io/resources/vors/rosenbrock.png
[rosenbrock]: https://en.wikipedia.org/wiki/Rosenbrock_function

### affine-2d

![Affine transformation][affine2d-jpg]

[affine2d-jpg]: https://mpizenberg.github.io/resources/vors/affine2d.jpg

_Original image on the left, automatically extracted "template" on the right._

In this example, we find the parameters of the affine transformation from
the extracted "template" image on the right to the original image on the left.
In the example we randomly generate the affine transformation,
represented by the following matrix in homogeneous coordinates:

```
[ 1+p1  p3  p5 ]
[  p2  1+p4 p6 ]
[  0    0   1  ]
```

We then optimize a direct image alignment problem of the form:

```
residual(x) = I(warp(x)) - T(x)
```

Where `T` is a template image,
`I` an transformed image by a warp function (that should align with our original image),
`x` a pixel coordinates in the template image,
`warp` a 2D affine transformation of the form described above.

Just like for the camera tracking, the resolution uses an inverse compositional approach
as described in [Baker and Matthews, 2001][baker],
with a Levenberg-Marquardt optimization algorithm.
It also uses a multi-scale approach, where the solution at one level serves
as an initialization for the next one.
This exercise on a 2D affine transformation was very useful before implementing
the actual 3D camera reprojection optimization.

You can run the example with the image of your choosing as follows:

```sh
cargo run --release --example optim_affine-2d /path/to/image
```
