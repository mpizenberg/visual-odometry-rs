extern crate nalgebra as na;
extern crate num_traits;
extern crate num_traits as num;

use na::{DMatrix, Scalar};
// use num::NumCast;
// use std::ops::{Add, Div};

pub fn pyramid<I, F, T, U>(mat: DMatrix<T>, init: I, f: F) -> Vec<DMatrix<U>>
where
    I: Fn(DMatrix<T>) -> DMatrix<U>,
    F: Fn(&DMatrix<U>) -> Option<DMatrix<U>>,
    T: Scalar,
    U: Scalar,
{
    let mut pyr = Vec::new();
    pyr.push(init(mat));
    while let Some(half_res) = f(pyr.last().unwrap()) {
        pyr.push(half_res);
    }
    pyr
}

pub fn halve<F, T, U>(mat: &DMatrix<T>, f: F) -> Option<DMatrix<U>>
where
    F: Fn(T, T, T, T) -> U,
    T: Scalar,
    U: Scalar,
    // U: Copy + NumCast + Add<U, Output = U> + Div<U, Output = U>,
{
    let (r, c) = mat.shape();
    let half_r = r / 2;
    let half_c = c / 2;
    if half_r == 0 || half_c == 0 {
        None
    } else {
        // let four: U = num::cast(4u8).unwrap();
        let half_mat = DMatrix::<U>::from_fn(half_r, half_c, |i, j| {
            let a = mat[(2 * i, 2 * j)];
            let b = mat[(2 * i + 1, 2 * j)];
            let c = mat[(2 * i, 2 * j + 1)];
            let d = mat[(2 * i + 1, 2 * j + 1)];
            // let a: U = num::cast(mat[(2 * i, 2 * j)]).unwrap();
            // let b: U = num::cast(mat[(2 * i + 1, 2 * j)]).unwrap();
            // let c: U = num::cast(mat[(2 * i, 2 * j + 1)]).unwrap();
            // let d: U = num::cast(mat[(2 * i + 1, 2 * j + 1)]).unwrap();
            // num::cast((a + b + c + d) / four).unwrap()
            f(a, b, c, d)
        });
        Some(half_mat)
    }
}
