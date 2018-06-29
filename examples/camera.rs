extern crate computer_vision_rs as cv;
extern crate nalgebra;

// use cv::camera::{Extrinsics, Intrinsics, Pinhole, Quaternions};
use nalgebra::{Matrix4, Projective3, Quaternion, RowVector4, UnitQuaternion};

fn main() {
    // let intrinsics = Intrinsics::Pinhole(Pinhole {
    //     principal_point: (319.5, 239.5),
    //     focal_length: 1.0,
    //     scaling: (481.20, 480.00),
    //     skew: 0.0,
    // });
    let proj_0 = Projective3::from_matrix_unchecked(Matrix4::from_rows(&[
        RowVector4::new(-0.999762, 0.0, -0.021799, 1.370500),
        RowVector4::new(0.0, 1.0, 0.0, 1.517390),
        RowVector4::new(0.021799, 0.0, -0.999762, 1.449630),
        RowVector4::new(0.0, 0.0, 0.0, 1.0),
    ]));
    let proj_1 = Projective3::from_matrix_unchecked(Matrix4::from_rows(&[
        RowVector4::new(-0.999738, -0.000418, -0.022848, 1.370020),
        RowVector4::new(-0.000464, 0.999998, 0.002027, 1.526344),
        RowVector4::new(0.022848, 0.002037, -0.999737, 1.448990),
        RowVector4::new(0.0, 0.0, 0.0, 1.0),
    ]));
    let proj_50 = Projective3::from_matrix_unchecked(Matrix4::from_rows(&[
        RowVector4::new(-0.980396, 0.089427, -0.175573, 1.244141),
        RowVector4::new(0.087771, 0.995992, 0.017191, 1.526088),
        RowVector4::new(0.176407, 0.001444, -0.984316, 1.457656),
        RowVector4::new(0.0, 0.0, 0.0, 1.0),
    ]));
    let proj_01 = proj_1 * proj_0.inverse();
    let proj_050 = proj_50 * proj_0.inverse();
    // println!("Projection 0 to 1: {}", proj_01.unwrap());
    println!("Projection 0 to 50: {}", proj_050.unwrap());
    let orientation_0 = UnitQuaternion::from_quaternion(Quaternion::new(1.0, 0.0, 0.0, 0.0));
    let orientation_1 = UnitQuaternion::from_quaternion(Quaternion::new(
        0.999999,
        -0.00101358,
        0.00052453,
        -0.000231475,
    ));
    let orientation_50 = UnitQuaternion::from_quaternion(Quaternion::new(
        0.995981,
        -0.00516676,
        0.0775787,
        0.0444655,
    ));
    let rotation_01 = orientation_0.rotation_to(&orientation_1);
    let rotation_050 = orientation_0.rotation_to(&orientation_50);
    // println!("Rotation 0 to 1: {}", rotation_01.to_rotation_matrix());
    println!("Rotation 0 to 50: {}", rotation_050.to_rotation_matrix());
}
