use nalgebra::{DMatrix, Scalar};

pub type Float = f32;

// Same as pyramid but limits the max number of levels.
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

// Half the resolution with the given function until it returns None.
// Consumes the matrix since we have an "equivalent" at the beginning of the pyramid.
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

// pub struct MultiResIter<F>
// where
//     F: FnMut(DMatrix<Float>) -> Option<DMatrix<Float>>,
// {
//     matrix: DMatrix<Float>,
//     next: F,
//     index: usize,
// }
//
//
// impl Iterator for MultiRes {
//     type Item = DMatrix<Float>;
//     fn next(&mut self) -> Option<DMatrix<Float>> {
//         match self.index {
//             0 => {Some(self.matrix)}
//         }

pub fn mean_pyramid(max_levels: usize, mat: DMatrix<u8>) -> Vec<DMatrix<Float>> {
    pyramid_with_max_n_levels(
        max_levels,
        mat,
        |m| m.map(|x| x as Float / 255.0),
        |m| halve(m, |a, b, c, d| (a + b + c + d) / 4.0),
    )
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

pub fn gradients(multires_mat: &Vec<DMatrix<Float>>) -> Vec<DMatrix<Float>> {
    let nb_levels = multires_mat.len();
    multires_mat
        .iter()
        .take(nb_levels - 1)
        .map(|mat| halve(mat, gradient_squared_norm).unwrap())
        .collect()
}

pub fn gradients_xy(multires_mat: &Vec<DMatrix<Float>>) -> Vec<(DMatrix<Float>, DMatrix<Float>)> {
    let nb_levels = multires_mat.len();
    multires_mat
        .iter()
        .take(nb_levels - 1)
        .map(|mat| {
            (
                halve(mat, gradient_x).unwrap(),
                halve(mat, gradient_y).unwrap(),
            )
        })
        .collect()
}

pub fn gradient_x(a: Float, b: Float, c: Float, d: Float) -> Float {
    (c + d - a - b) / 2.0
}

pub fn gradient_y(a: Float, b: Float, c: Float, d: Float) -> Float {
    (b - a + d - c) / 2.0
}

fn gradient_squared_norm(a: Float, b: Float, c: Float, d: Float) -> Float {
    let dx = c + d - a - b;
    let dy = b - a + d - c;
    (dx * dx + dy * dy) / 4.0
}
