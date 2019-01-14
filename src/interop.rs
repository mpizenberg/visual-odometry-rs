use image::{GrayImage, ImageBuffer, Luma, Rgb, RgbImage};
use nalgebra::DMatrix;

pub fn image_from_matrix(mat: &DMatrix<u8>) -> GrayImage {
    let (nb_rows, nb_cols) = mat.shape();
    let mut img_buf = GrayImage::new(nb_cols as u32, nb_rows as u32);
    for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
        *pixel = Luma([mat[(y as usize, x as usize)]]);
    }
    img_buf
}

pub fn rgb_from_matrix(mat: &DMatrix<(u8, u8, u8)>) -> RgbImage {
    let (nb_rows, nb_cols) = mat.shape();
    let mut img_buf = RgbImage::new(nb_cols as u32, nb_rows as u32);
    for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
        let (r, g, b) = mat[(y as usize, x as usize)];
        *pixel = Rgb([r, g, b]);
    }
    img_buf
}

// Use a borrowed reference to the matrix buffer.
// Due to a difference of row major instead of column major,
// this produces a mirrored + rotated image (transposed image).
pub fn image_from_matrix_transposed(mat: &DMatrix<u8>) -> ImageBuffer<Luma<u8>, &[u8]> {
    let (nb_rows, nb_cols) = mat.shape();
    ImageBuffer::from_raw(nb_rows as u32, nb_cols as u32, mat.as_slice())
        .expect("Buffer not large enough")
}

pub fn matrix_from_image(img: GrayImage) -> DMatrix<u8> {
    let (width, height) = img.dimensions();
    DMatrix::from_row_slice(height as usize, width as usize, &img.into_raw())
}
