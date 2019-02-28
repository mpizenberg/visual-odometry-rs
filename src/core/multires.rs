use crate::core::gradient;
use nalgebra::{DMatrix, Scalar};

pub type Float = f32;

// Recursively generate a pyramid of matrices where each level
// is half the previous resolution, computed with the mean of each 2x2 block.
pub fn mean_pyramid(max_levels: usize, mat: DMatrix<u8>) -> Vec<DMatrix<u8>> {
    limited_sequence(
        max_levels,
        mat,
        |m| m,
        |m| {
            halve(m, |a, b, c, d| {
                let a = a as u16;
                let b = b as u16;
                let c = c as u16;
                let d = d as u16;
                ((a + b + c + d) / 4) as u8
            })
        },
    )
}

// Recursively apply a function transforming the image
// until it's not possible anymore or the max number of iterations is reached.
// Using iterations = 0 has the same effect than iterations = 1 since it always has
// at least one matrix (the init matrix).
pub fn limited_sequence<I, F, T, U>(
    iterations: usize,
    mat: DMatrix<T>,
    init: I,
    f: F,
) -> Vec<DMatrix<U>>
where
    I: Fn(DMatrix<T>) -> DMatrix<U>,
    F: Fn(&DMatrix<U>) -> Option<DMatrix<U>>,
    T: Scalar,
    U: Scalar,
{
    let mut iteration = 1;
    let f_limited = |x: &DMatrix<U>| {
        if iteration < iterations {
            iteration = iteration + 1;
            f(x)
        } else {
            None
        }
    };
    sequence(mat, init, f_limited)
}

// Recursively apply a function transforming the image
// until it's not possible anymore.
pub fn sequence<I, F, T, U>(mat: DMatrix<T>, init: I, mut f: F) -> Vec<DMatrix<U>>
where
    I: Fn(DMatrix<T>) -> DMatrix<U>,
    F: FnMut(&DMatrix<U>) -> Option<DMatrix<U>>,
    T: Scalar,
    U: Scalar,
{
    let mut pyr = Vec::new();
    pyr.push(init(mat));
    while let Some(new_mat) = f(pyr.last().unwrap()) {
        pyr.push(new_mat);
    }
    pyr
}

// Halve the resolution of a matrix by applying a function to each 2x2 block.
// If one size of the matrix is < 2 then this function returns None.
// If one size is odd, its last line/column is dropped.
pub fn halve<F, T, U>(mat: &DMatrix<T>, f: F) -> Option<DMatrix<U>>
where
    F: Fn(T, T, T, T) -> U,
    T: Scalar,
    U: Scalar,
{
    let (r, c) = mat.shape();
    let half_r = r / 2;
    let half_c = c / 2;
    if half_r == 0 || half_c == 0 {
        None
    } else {
        let half_mat = DMatrix::<U>::from_fn(half_r, half_c, |i, j| {
            let a = mat[(2 * i, 2 * j)];
            let b = mat[(2 * i + 1, 2 * j)];
            let c = mat[(2 * i, 2 * j + 1)];
            let d = mat[(2 * i + 1, 2 * j + 1)];
            f(a, b, c, d)
        });
        Some(half_mat)
    }
}

// Gradients stuff ###################################################

// Compute centered gradients norm at each resolution from
// the image at the higher resolution.
// As a consequence their is one less level in the gradients pyramid.
pub fn gradients_squared_norm(multires_mat: &Vec<DMatrix<u8>>) -> Vec<DMatrix<u16>> {
    let nb_levels = multires_mat.len();
    multires_mat
        .iter()
        .take(nb_levels - 1)
        .map(|mat| halve(mat, gradient::bloc_squared_norm).unwrap())
        .collect()
}

// Compute centered gradients at each resolution from
// the image at the higher resolution.
// As a consequence their is one less level in the gradients pyramid.
pub fn gradients_xy(multires_mat: &Vec<DMatrix<u8>>) -> Vec<(DMatrix<i16>, DMatrix<i16>)> {
    let nb_levels = multires_mat.len();
    multires_mat
        .iter()
        .take(nb_levels - 1)
        .map(|mat| {
            (
                halve(mat, gradient::bloc_x).unwrap(),
                halve(mat, gradient::bloc_y).unwrap(),
            )
        })
        .collect()
}
