// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Helper function to compute gradients

use nalgebra as na;

/// Compute a centered gradient.
///
/// 1/2 * ( img(i+1,j) - img(i-1,j), img(i,j+1) - img(i,j-1) )
///
/// Gradients of pixels at the border of the image are set to 0.
#[allow(clippy::similar_names)]
pub fn centered(img: &na::DMatrix<u8>) -> (na::DMatrix<i16>, na::DMatrix<i16>) {
    // TODO: might be better to return DMatrix<(i16,i16)>?
    let (nb_rows, nb_cols) = img.shape();
    let top = img.slice((0, 1), (nb_rows - 2, nb_cols - 2));
    let bottom = img.slice((2, 1), (nb_rows - 2, nb_cols - 2));
    let left = img.slice((1, 0), (nb_rows - 2, nb_cols - 2));
    let right = img.slice((1, 2), (nb_rows - 2, nb_cols - 2));
    let mut grad_x = na::DMatrix::zeros(nb_rows, nb_cols);
    let mut grad_y = na::DMatrix::zeros(nb_rows, nb_cols);
    let mut grad_x_inner = grad_x.slice_mut((1, 1), (nb_rows - 2, nb_cols - 2));
    let mut grad_y_inner = grad_y.slice_mut((1, 1), (nb_rows - 2, nb_cols - 2));
    for j in 0..nb_cols - 2 {
        for i in 0..nb_rows - 2 {
            grad_x_inner[(i, j)] = (i16::from(right[(i, j)]) - i16::from(left[(i, j)])) / 2;
            grad_y_inner[(i, j)] = (i16::from(bottom[(i, j)]) - i16::from(top[(i, j)])) / 2;
        }
    }
    (grad_x, grad_y)
}

/// Compute squared gradient norm from x and y gradient matrices.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn squared_norm(grad_x: &na::DMatrix<i16>, grad_y: &na::DMatrix<i16>) -> na::DMatrix<u16> {
    grad_x.zip_map(grad_y, |gx, gy| {
        let gx = i32::from(gx);
        let gy = i32::from(gy);
        (gx * gx + gy * gy) as u16
    })
}

/// Compute squared gradient norm directly from the image.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn squared_norm_direct(im: &na::DMatrix<u8>) -> na::DMatrix<u16> {
    let (nb_rows, nb_cols) = im.shape();
    let top = im.slice((0, 1), (nb_rows - 2, nb_cols - 2));
    let bottom = im.slice((2, 1), (nb_rows - 2, nb_cols - 2));
    let left = im.slice((1, 0), (nb_rows - 2, nb_cols - 2));
    let right = im.slice((1, 2), (nb_rows - 2, nb_cols - 2));
    let mut squared_norm_mat = na::DMatrix::zeros(nb_rows, nb_cols);
    let mut grad_inner = squared_norm_mat.slice_mut((1, 1), (nb_rows - 2, nb_cols - 2));
    for j in 0..nb_cols - 2 {
        for i in 0..nb_rows - 2 {
            let gx = i32::from(right[(i, j)]) - i32::from(left[(i, j)]);
            let gy = i32::from(bottom[(i, j)]) - i32::from(top[(i, j)]);
            grad_inner[(i, j)] = ((gx * gx + gy * gy) / 4) as u16;
        }
    }
    squared_norm_mat
}

// BLOCS 2x2 ###################################################################

/// Horizontal gradient in a 2x2 pixels block.
///
/// The block is of the form:
///   a c
///   b d
pub fn bloc_x(a: u8, b: u8, c: u8, d: u8) -> i16 {
    let a = i16::from(a);
    let b = i16::from(b);
    let c = i16::from(c);
    let d = i16::from(d);
    (c + d - a - b) / 2
}

/// Vertical gradient in a 2x2 pixels block.
///
/// The block is of the form:
///   a c
///   b d
pub fn bloc_y(a: u8, b: u8, c: u8, d: u8) -> i16 {
    let a = i16::from(a);
    let b = i16::from(b);
    let c = i16::from(c);
    let d = i16::from(d);
    (b - a + d - c) / 2
}

/// Gradient squared norm in a 2x2 pixels block.
///
/// The block is of the form:
///   a c
///   b d
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn bloc_squared_norm(a: u8, b: u8, c: u8, d: u8) -> u16 {
    let a = i32::from(a);
    let b = i32::from(b);
    let c = i32::from(c);
    let d = i32::from(d);
    let dx = c + d - a - b;
    let dy = b - a + d - c;
    // I have checked that the max value is in u16.
    ((dx * dx + dy * dy) / 4) as u16
}
