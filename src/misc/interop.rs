// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Interoperability conversions between the image and matrix types.

use image::{GrayImage, ImageBuffer, Luma, Rgb, RgbImage};
use nalgebra::DMatrix;

/// Convert an `u8` matrix into a `GrayImage`.
/// Inverse operation of `matrix_from_image`.
///
/// Performs a transposition to accomodate for the
/// column major matrix into the row major image.
#[allow(clippy::cast_possible_truncation)]
pub fn image_from_matrix(mat: &DMatrix<u8>) -> GrayImage {
    let (nb_rows, nb_cols) = mat.shape();
    let mut img_buf = GrayImage::new(nb_cols as u32, nb_rows as u32);
    for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
        *pixel = Luma([mat[(y as usize, x as usize)]]);
    }
    img_buf
}

/// Convert an `(u8,u8,8)` matrix into an `RgbImage`.
///
/// Performs a transposition to accomodate for the
/// column major matrix into the row major image.
#[allow(clippy::cast_possible_truncation)]
pub fn rgb_from_matrix(mat: &DMatrix<(u8, u8, u8)>) -> RgbImage {
    let (nb_rows, nb_cols) = mat.shape();
    let mut img_buf = RgbImage::new(nb_cols as u32, nb_rows as u32);
    for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
        let (r, g, b) = mat[(y as usize, x as usize)];
        *pixel = Rgb([r, g, b]);
    }
    img_buf
}

/// Create a gray image with a borrowed reference to the matrix buffer.
///
/// Very performant since no copy is performed,
/// but produces a transposed image due to differences in row/column major.
#[allow(clippy::cast_possible_truncation)]
pub fn image_from_matrix_transposed(mat: &DMatrix<u8>) -> ImageBuffer<Luma<u8>, &[u8]> {
    let (nb_rows, nb_cols) = mat.shape();
    ImageBuffer::from_raw(nb_rows as u32, nb_cols as u32, mat.as_slice())
        .expect("Buffer not large enough")
}

/// Convert a `GrayImage` into an `u8` matrix.
/// Inverse operation of `image_from_matrix`.
pub fn matrix_from_image(img: GrayImage) -> DMatrix<u8> {
    let (width, height) = img.dimensions();
    DMatrix::from_row_slice(height as usize, width as usize, &img.into_raw())
}
