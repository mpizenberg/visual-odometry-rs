extern crate computer_vision_rs as cv;
extern crate nalgebra;

use cv::camera::{Camera, Float};
use cv::icl_nuim::{self, INTRINSICS};
use nalgebra::Point2;

fn main() {
    // Use bottom left corner of the image in frames 1, 80, 90 and 240
    compute_3d_coordinates(1, 268, 393);
    compute_3d_coordinates(80, 263, 130);
    compute_3d_coordinates(90, 254, 82);
    compute_3d_coordinates(240, 206, 605);
}

fn compute_3d_coordinates(frame_number: usize, line: usize, column: usize) -> () {
    // Retrieve depth map (as an u16 matrix)
    let (_, depth_map) = icl_nuim::open_imgs(frame_number).unwrap();

    // Coordinates (x = column, y = line) and depth of pixel
    let xy = Point2::new(column as Float, line as Float);
    let z = depth_map[(line, column)] as Float / icl_nuim::DEPTH_SCALE;

    // Read trajectory file with TUM RGBD syntax (same as icl nuim syntax)
    let all_extrinsics = icl_nuim::read_extrinsics("data/trajectory-gt.txt").unwrap();

    // Create camera
    // WARNING!!! extrinsics indices have an offset
    let camera = Camera::new(INTRINSICS.clone(), all_extrinsics[frame_number - 1]);

    // Print 3D coordinates of point computed with cameras and depth info
    println!(
        "3D point coordinates according to camera {}: {}",
        frame_number,
        camera.back_project(xy, z)
    );
}
