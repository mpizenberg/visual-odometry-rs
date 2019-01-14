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

### camera.rs

Use the bottom left corner of the image on the wall
to validate camera projections.

The corner coordinates have been manually retrieved in 4 frames (1, 80, 90, 240).
Back projections (to 3D world coordinates) of each point are displayed to check that
they correspond to a unique 3D point in the scene.

> This example brought to light the fact that the first line in the camera
> ground truth file (`data/trajectory-gt.txt`) corresponds to the **second**
> frame, i.e. the frame 1 (first frame being number 0).
> That is why there are 1509 images, but only 1508 camera coordinates in the file.

### depth_map_candidates.rs

Compute and visualize multi-resolution inverse depth maps of candidates points.
