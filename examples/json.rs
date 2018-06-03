#[macro_use]
extern crate serde_derive;
extern crate serde;
extern crate serde_json;

use std::fs::File;
use std::path::Path;

#[allow(dead_code)]
fn main() {
    // example camera instance
    let camera = CameraIntrinsics::PinholeWithDistortion(
        Pinhole {
            principal_point: (320.0, 240.0),
            focal_length: 1.0,
            scaling: (2.0, 3.0),
            skew: 0.0,
        },
        RadialTangential {
            radial_coeffs: (0.0, 0.0, 0.0),
            tangential_coeffs: (0.0, 0.0),
        },
    );

    // json file supposed to contain the same camera instance
    let json_file_path = Path::new("data/camera.json");
    let json_file = File::open(json_file_path).expect("file not found");
    let deserialized_camera: CameraIntrinsics =
        serde_json::from_reader(json_file).expect("error while reading json");

    // verify that deserialized_camera equals camera
    assert_eq!(camera, deserialized_camera);
}

type Float = f32;

#[derive(Serialize, Deserialize, PartialEq, Debug)]
enum CameraIntrinsics {
    Pinhole(Pinhole),
    PinholeWithDistortion(Pinhole, RadialTangential),
    FishEye(FishEye),
    Fov(Fov),
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct Pinhole {
    principal_point: (Float, Float),
    focal_length: Float,
    scaling: (Float, Float),
    skew: Float,
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct RadialTangential {
    radial_coeffs: (Float, Float, Float),
    tangential_coeffs: (Float, Float),
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct FishEye {}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct Fov {}
