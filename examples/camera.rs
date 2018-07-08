extern crate computer_vision_rs as cv;
extern crate nalgebra;

use cv::camera::{Camera, Extrinsics, Float, Intrinsics};
use nalgebra::{Isometry3, Point2, Point3, Quaternion, Translation3, UnitQuaternion};

fn main() {
    // Camera intrinsics.
    let intrinsics = Intrinsics {
        principal_point: (319.5, 239.5),
        focal_length: 1.0,
        scaling: (481.20, -480.00),
        skew: 0.0,
    };

    // Camera at frame 1.
    let translation_1 = Translation3::new(0.0, 0.0, -2.25);
    let rotation_1 = UnitQuaternion::from_quaternion(Quaternion::new(1.0, 0.0, 0.0, 0.0));
    let extrinsics_1 = Extrinsics::new(translation_1, rotation_1);
    let camera_1 = Camera::new(intrinsics.clone(), extrinsics_1);

    // Camera at frame 600.
    let translation_600 = Translation3::new(0.310245, -0.43235, -1.48106);
    let rotation_600 =
        UnitQuaternion::from_quaternion(Quaternion::new(0.933403, 0.0471923, 0.322162, 0.150811));
    let extrinsics_600 = Extrinsics::new(translation_600, rotation_600);
    let camera_600 = Camera::new(intrinsics.clone(), extrinsics_600);

    // Example showing that Extrinsics is similar to nalgebra Isometry type.
    let iso_600 = Isometry3::from_parts(translation_600, rotation_600);
    println!("iso {}", iso_600 * Point3::new(1.0, 2.0, 3.0));
    println!(
        "project {}",
        camera_600.extrinsics.project(Point3::new(1.0, 2.0, 3.0))
    );
    println!(
        "back project {}",
        camera_600
            .extrinsics
            .back_project(Point3::new(1.0, 2.0, 3.0))
    );

    // Bottom left picture frame corner in frame 1.
    let bl_pos_1 = Point2::new(386.0, 277.0);
    let bl_depth_1 = 16820.0 / 5000.0;

    // Bottom left picture frame corner in frame 600.
    let bl_pos_600 = Point2::new(14.0, 102.0);
    let bl_depth_600 = 10670.0 / 5000.0;

    // Back project the bottom left corner in both frame 1 and 600.
    // We should normally obtain the same 3D point.
    same_3d_back_projected(
        (bl_pos_1, bl_depth_1),
        (bl_pos_600, bl_depth_600),
        &camera_1,
        &camera_600,
    );

    // Bottom right picture frame corner in frame 1.
    let br_pos_1 = Point2::new(556.0, 276.0);
    let br_depth_1 = 16990.0 / 5000.0;

    // Bottom right picture frame corner in frame 600.
    let br_pos_600 = Point2::new(242.0, 188.0);
    let br_depth_600 = 14500.0 / 5000.0;

    // Back project the bottom right corner in both frame 1 and 600.
    // We should normally obtain the same 3D point.
    same_3d_back_projected(
        (br_pos_1, br_depth_1),
        (br_pos_600, br_depth_600),
        &camera_1,
        &camera_600,
    );

    // At half resolution.
    println!("\nHalf resolution:");
    let bl_pos_1_half = 0.5 * bl_pos_1;
    let bl_pos_600_half = 0.5 * bl_pos_600;
    same_3d_back_projected(
        (bl_pos_1_half, bl_depth_1),
        (bl_pos_600_half, bl_depth_600),
        &camera_1.half_res(),
        &camera_600.half_res(),
    );
}

fn same_3d_back_projected(
    point1: (Point2<Float>, Float),
    point2: (Point2<Float>, Float),
    camera1: &Camera,
    camera2: &Camera,
) -> () {
    println!(
        "Back projection in first frame: {}",
        camera1.back_project(point1.0, point1.1)
    );
    println!(
        "Back projection in second frame: {}",
        camera2.back_project(point2.0, point2.1)
    );
}
