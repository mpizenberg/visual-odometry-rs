use nalgebra::{DMatrix, Scalar};

pub fn pyramid_with_max_n_levels<I, F, T, U>(
    levels: usize,
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
    let mut level = 1;
    let new_f = |x: &DMatrix<U>| {
        if level < levels {
            level = level + 1;
            f(x)
        } else {
            None
        }
    };
    pyramid(mat, init, new_f)
}

pub fn mean_pyramid(max_levels: usize, mat: DMatrix<u8>) -> Vec<DMatrix<u8>> {
    pyramid_with_max_n_levels(
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

pub fn pyramid<I, F, T, U>(mat: DMatrix<T>, init: I, mut f: F) -> Vec<DMatrix<U>>
where
    I: Fn(DMatrix<T>) -> DMatrix<U>,
    F: FnMut(&DMatrix<U>) -> Option<DMatrix<U>>,
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

pub fn gradients(multires_mat: &Vec<DMatrix<u8>>) -> Vec<DMatrix<u16>> {
    let nb_levels = multires_mat.len();
    multires_mat
        .iter()
        .take(nb_levels - 1)
        .map(|mat| halve(mat, gradient_squared_norm).unwrap())
        .collect()
}

fn gradient_squared_norm(a: u8, b: u8, c: u8, d: u8) -> u16 {
    let a = a as i32;
    let b = b as i32;
    let c = c as i32;
    let d = d as i32;
    let dx = c + d - a - b;
    let dy = b - a + d - c;
    // I have checked that the max value is in u16.
    ((dx * dx + dy * dy) / 4) as u16
}
