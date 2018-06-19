extern crate image;
extern crate nalgebra as na;

use self::image::{GrayImage, Luma};
use self::na::DMatrix;

pub fn image_from_matrix(mat: &DMatrix<u8>) -> GrayImage {
    let (nb_rows, nb_cols) = mat.shape();
    let mut img_buf = GrayImage::new(nb_cols as u32, nb_rows as u32);
    for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
        *pixel = Luma([mat[(y as usize, x as usize)]]);
    }
    img_buf
}

pub fn matrix_from_image(img: GrayImage) -> DMatrix<u8> {
    let (width, height) = img.dimensions();
    DMatrix::from_row_slice(height as usize, width as usize, &img.into_raw())
}
