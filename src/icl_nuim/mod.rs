extern crate image;
extern crate nalgebra;

use camera::{Camera, Extrinsics, Intrinsics};
use helper;
use interop;
use multires;

use nalgebra::DMatrix;

const INTRINSICS: Intrinsics = Intrinsics {
    principal_point: (319.5, 239.5),
    focal_length: 1.0,
    scaling: (481.20, -480.00),
    skew: 0.0,
};

pub fn prepare_data(
    id: usize,
    extrinsics: &Vec<Extrinsics>,
) -> Result<(Vec<Camera>, Vec<DMatrix<u8>>, DMatrix<u16>), image::ImageError> {
    let (img, depth) = open_imgs(id)?;
    Ok((
        Camera::new(INTRINSICS.clone(), extrinsics[id - 1].clone()).multi_res(6),
        multires::mean_pyramid(6, img),
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
