# Computer Vision Experiments in Rust

In this repository, I experiment on computer vision in Rust.
My current focus is direct visual odometry.
You can read [DSO][dso] for more context about this technique.

[dso]: https://github.com/JakobEngel/dso

## Examples

Examples are made along the way and located in the `examples/` directory.
They often rely on images from datasets not included in this repository.
Mainly images from the first sequence of the [ICL-NUIM][icl-nuim] dataset.

[icl-nuim]: https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html

### 01-camera.rs

Use the bottom left corner of the image on the wall
to validate camera projections.

The corner coordinates have been manually retrieved in 4 frames (1, 80, 90, 240).
Back projections (to 3D world coordinates) of each point are displayed to check that
they correspond to a unique 3D point in the scene.

> This example brought to light the fact that the first line in the camera
> ground truth file (`data/trajectory-gt.txt`) corresponds to the **second**
> frame, i.e. the frame 1 (first frame being number 0).
> That is why there are 1509 images, but only 1508 camera coordinates in the file.

### 02-depth_map_candidates.rs

Compute and visualize multi-resolution inverse depth maps of candidates points.

### 03-multires_depth_strategies.rs

The purpose is to evaluate different strategies to construct the multi-resolution
depth map from the highest resolution.
In this example, we evaluate two strategies:

1. `inverse_depth::strategy_statistically_similar`
2. `inverse_depth::strategy_dso_mean`

The `dso_mean` strategy is the one employed by DSO.
The rule is simple, if at least one of the 4 subpixels of the higher resolution
has a known inverse depth,
we compute the mean of the subpixels inverse depths.

The `statistically_similar` strategy consists in merging pixels
only if the follow some statistic rules.
Otherwise, the point is discarded for the lower resolution.

Results are saved into a CSV file,
which can be explored using a tool like [data voyager][data-voyager].

At the moment of writing this test, no significant difference is visible
between the two strategies, especially with a reprojection
with less than 5 frames difference.
This may also be due to the case that most of the time,
there are only two points to merge, and it behaves like the `dso_mean` case.

[data-voyager]: http://vega.github.io/voyager/

### 04-optim_exp.rs

This is the first example of a series on optimization.
The purpose is to test implementations of non-linear optimization algorithms
on simple cases, before trying it on camera transformations.

In this first example, we estimate a noisy data modeled by the function

```
f: a, x -> exp(-a*x).
```

We use the API provided by the `optimization` module to estimate
the parameter `a` with two algorithms.
The first is the Gauss-Newton (GN) algorithm,
the second is the Levenberg-Marquardt (LM) algorithm.
Both are quasi-Newton algorithms in the sense that they use
a first order approximation of the hessian.

The LM algorithm actually is a slight variation of GN.
The difference being that the diagonal of the hessian matrix
is modified dynamically with a coefficient,
transforming it into a hybrid between GN and gradient descent.
It has better convergence properties than GN when farrer from the solution.

### 04-optim_exp_bis.rs

Same as `04-optim_exp`.
The only difference is that the API used comes from the module `optimization_bis`.
This second API is, in my opinion, more versatile,
by giving more control to the caller.
In particular, it enables, avoiding computations of derivatives
when the new model gives worst residuals.

### 05-optim_rosenbrock.rs

Use the `bis` optimization API to find the minimum of the Rosenbrock function:

```
f: a,b,x,y -> (a-x)^2 + b * (y - x^2)^2
```

This function is often used as a benchmark for optimization algorithms,
so it made sense to verify that our implementation was versatile enough to solve it.
