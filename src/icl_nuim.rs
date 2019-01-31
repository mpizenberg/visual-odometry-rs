use crate::camera::{self, Camera, Extrinsics, Intrinsics};
use crate::helper;
use crate::interop;
use crate::multires_float;

use nalgebra::{DMatrix, Quaternion, Translation3, UnitQuaternion};
use std::fs::File;
use std::io;
use std::io::prelude::Read;

pub type Float = f32;

// U16 depth values are scaled for better precision.
// So 5000 in the 16 bits gray png corresponds to 1m.
pub const DEPTH_SCALE: Float = 5000.0;

pub const INTRINSICS: Intrinsics = Intrinsics {
    principal_point: (319.5, 239.5),
    focal_length: 1.0,
    scaling: (481.20, -480.00),
    skew: 0.0,
};

pub fn prepare_data(
    id: usize,
    extrinsics: &Vec<Extrinsics>,
) -> Result<(Vec<Camera>, Vec<DMatrix<Float>>, DMatrix<u16>), image::ImageError> {
    let (img, depth) = open_imgs(id)?;
    Ok((
        Camera::new(INTRINSICS.clone(), extrinsics[id - 1].clone()).multi_res(6),
        multires_float::mean_pyramid(6, img),
        depth,
    ))
}

pub fn open_imgs(id: usize) -> Result<(DMatrix<u8>, DMatrix<u16>), image::ImageError> {
    let img_mat =
        interop::matrix_from_image(image::open(&format!("icl-rgb/{}.png", id))?.to_luma());
    let (w, h, buffer) = helper::read_png_16bits(&format!("icl-depth/{}.png", id))?;
    let depth_map = DMatrix::from_row_slice(h, w, buffer.as_slice());
    Ok((img_mat, depth_map))
}

pub fn read_extrinsics(file_path: &str) -> Result<Vec<Extrinsics>, io::Error> {
    let mut file_content = String::new();
    File::open(file_path)?.read_to_string(&mut file_content)?;
    let mut extrinsics = Vec::new();
    for line in file_content.lines() {
        let values: Vec<Float> = line.split(' ').filter_map(|s| s.parse().ok()).collect();
        assert_eq!(8, values.len(), "There was an issue in parsing:\n{}", line);
        let translation = Translation3::new(values[1], values[2], values[3]);
        let rotation = UnitQuaternion::from_quaternion(Quaternion::new(
            values[7], values[4], values[5], values[6],
        ));
        extrinsics.push(camera::extrinsics::from_parts(translation, rotation));
    }
    Ok(extrinsics)
}
