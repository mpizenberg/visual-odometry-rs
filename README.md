# Visual Odometry in Rust (vors)

This repository provides both a library ("crate" as we say in Rust) named visual-odometry-rs,
(shortened vors) and a binary program named `vors_track`,
for camera tracking ("visual odometry").

The program works on datasets following the [TUM RGB-D dataset format][tum-rgbd].
It is roughly a hundred lines of code (see `src/bin/vors_track.rs`),
built upon the visual-odometry-rs crate also provided here.

Once you have cloned this repository,
you can run the binary program `vors_track` with cargo directly as follows:

```sh
cargo run --release --bin vors_track -- fr1 /path/to/some/freiburg1/dataset/associations.txt
```

Have a look at [mpizenberg/rgbd-tracking-evaluation][rgbd-track-eval]
for more info about the dataset requirements to run the binary program `vors_track`.

The library is organized around four base namespaces:

- `core::` Core modules for computing gradients, candidate points, camera tracking etc.
- `dataset::` Helper modules for handling specific datasets.
  Currently only provides a module for TUM RGB-D compatible datasets.
- `math::` Basic math modules for functionalities not already provided by [nalgebra][nalgebra],
  like Lie algebra for so3, se3, and an iterative optimizer trait.
- `misc::` Helper modules for interoperability, visualization, and other things that did
  not fit elsewhere yet.

[tum-rgbd]: https://vision.in.tum.de/data/datasets/rgbd-dataset
[rgbd-track-eval]: https://github.com/mpizenberg/rgbd-tracking-evaluation
[nalgebra]: https://www.nalgebra.org/

## Library Usage Examples

Self contained examples for usage of the API are available in the `examples/` directory.
A readme is also present there for more detailed explanations on these examples.

## Functionalities and Vision

Currently, vors provides a **visual odometry framework for working on direct RGB-D camera tracking**.
Setting all this from the ground up took a lot of time and effort,
but I think it is mature enough to be shared as is now.
Beware, however, that the API is evolving a lot.
My hope is that in the near future, we can improve the reach of this project
by working both on research extensions, and platform availability.

Example research extensions:

- Using disparity search for depth initialization to be compatible with RGB (no depth) camera.
- Adding a photometric term to the residual to account for automatic exposure variations.
- Adding automatic photometric and/or geometric camera calibration.
- Building a sliding window of keyframes optimization as in [DSO][dso] to reduce drift.
- Intregrating loop closure and pose graph optimization for having a robust vSLAM system.
- Fusion with IMU for improved tracking and reducing scale drift.
- Modelization of rolling shutter (in most cameras) into the optimization problem.
- Extension to stereo cameras.
- Extension to omnidirectional cameras.

Example platform extensions:

- Making a C FFI to be able to run on systems with C drivers (kinect, realsense, ...).
- Porting to the web with WebAssembly.
- Porting to ARM for running in embedded systems and phones.

## Background Story

Initially, this repository served as a personal experimental sandbox for computer vision in Rust.
See for example my original questions on the rust [discourse][discourse] and [reddit channel][reddit].
Turns out I struggled a bit at first but then really liked the Rust way, compared to C++.

As the name suggests, the focus is now on [visual odometry][vo],
specifically on the recent research field of direct visual odometry.
A reasonable introduction is available in those [lecture slides][vo-slides]
by Waterloo Autonomous Vehicles lab.

In particular, this project initially aimed at improving on the work of [DSO][dso]
by J. Engel et. al. but with all the advantages of using the [Rust programming language][rust],
including:

- Performance without sacrificing code readability
- No memory error, and much higher code safety and reliability
- Friendly tooling ecosystem, no dependency issues, basically one-liner compilation and run
- Best tooling for porting to the web with WebAssembly
- Growing and mindful resources for porting to embedded systems
- Wonderful community

[discourse]: https://users.rust-lang.org/t/computer-vision-in-rust/16198
[reddit]: https://www.reddit.com/r/rust/comments/84s5zo/computer_vision_in_rust/
[vo]: https://en.wikipedia.org/wiki/Visual_odometry
[vo-slides]: http://wavelab.uwaterloo.ca/slam/2017-SLAM/Lecture14-Direct_visual_inertial_odometry_and_SLAM/slides.pdf
[dso]: https://github.com/JakobEngel/dso
[rust]: https://www.rust-lang.org/

## License (MPL-2.0)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
