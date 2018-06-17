extern crate image;
extern crate nalgebra as na;
extern crate num_traits;
extern crate num_traits as num;

mod interop;
mod multires;

use na::DMatrix;

const NB_PIXELS_THRESH: usize = 400;

// #[allow(dead_code)]
fn main() {
    // Load a color image and transform into grayscale.
    let img = image::open("icl-rgb/0.png")
        .expect("Cannot open image")
        .to_luma();

    // Create an equivalent matrix.
    let img_matrix = interop::matrix_from_image(img);

    // Compute pyramid of matrices.
    let multires_img = multires::pyramid(
        img_matrix,
        |mat| mat,
        |mat| halve_with_thresh(NB_PIXELS_THRESH, mat),
    );

    // Compute pyramid of gradients (without first level).
    let nb_levels = multires_img.len();
    let multires_gradient_norm: Vec<DMatrix<u16>> = multires_img
        .iter()
        .take(nb_levels - 1)
        .map(half_gradient_norm)
        .collect();

    // // Save pyramid of images.
    // multires_img.iter().enumerate().for_each(|(i, mat)| {
    //     let out_name = &["out/img_", i.to_string().as_str(), ".png"].concat();
    //     interop::image_from_matrix(&mat).save(out_name).unwrap();
    // });
    //
    // // Save pyramid of gradients.
    // multires_gradient_norm
    //     .iter()
    //     .enumerate()
    //     .for_each(|(i, mat)| {
    //         let out_name = &["out/gradient_norm_", (i + 1).to_string().as_str(), ".png"].concat();
    //         interop::image_from_matrix(&mat.map(|x| (x as f32).sqrt() as u8))
    //             .save(out_name)
    //             .unwrap();
    //     });

    let smaller = &multires_gradient_norm.last().unwrap();
    let (nrows, ncols) = smaller.shape();
    let premask = DMatrix::repeat(nrows, ncols, true);
    let second_grad = &multires_gradient_norm[nb_levels - 3];
    let test = bloc_2x2_filter(&premask, second_grad, higher_than_mean_with_thresh);
    // save test
    interop::image_from_matrix(&test.map(|x| if x { 255u8 } else { 0u8 }))
        .save("out/test.png")
        .unwrap();

    // canditates
    let point_candidates = candidates(&multires_gradient_norm);
    interop::image_from_matrix(&point_candidates.map(|x| if x { 255u8 } else { 0u8 }))
        .save("out/candidates.png")
        .unwrap();
}

fn candidates(gradients: &Vec<DMatrix<u16>>) -> DMatrix<bool> {
    let (nrows, ncols) = gradients.last().unwrap().shape();
    let pre_mask = DMatrix::repeat(nrows, ncols, true);
    let mask = gradients
        .iter()
        .rev() // start with lower res
        .skip(1) // skip lower since all points are good
        .fold(pre_mask.clone(), |mask_acc, grad_mat| {
            bloc_2x2_filter(&mask_acc, &grad_mat, higher_than_mean_with_thresh)
        });
    mask
}

fn higher_than_mean_with_thresh(a: u16, b: u16, c: u16, d: u16) -> [bool; 4] {
    let thresh = 7;
    // let mean = (a as u32 + b as u32 + c as u32 + d as u32) / 4;
    let mut temp = [(a, 0usize), (b, 1usize), (c, 2usize), (d, 3usize)];
    temp.sort_unstable_by(|(x, _), (y, _)| x.cmp(y));
    let (_, first) = temp[3];
    let (x, second) = temp[2];
    let (y, _) = temp[1];
    let mut result = [false, false, false, false];
    result[first] = true;
    if x > y + thresh {
        result[second] = true;
    }
    result
}

fn bloc_2x2_filter<F>(pre_mask: &DMatrix<bool>, mat: &DMatrix<u16>, f: F) -> DMatrix<bool>
where
    F: Fn(u16, u16, u16, u16) -> [bool; 4],
{
    let (nrows, ncols) = mat.shape();
    let (nrows_2, ncols_2) = pre_mask.shape();
    assert_eq!((nrows_2, ncols_2), (nrows / 2, ncols / 2));
    let mut mask = DMatrix::repeat(nrows, ncols, false);
    for j in 0..(ncols_2) {
        for i in 0..(nrows_2) {
            if pre_mask[(i, j)] {
                let a = mat[(2 * i, 2 * j)];
                let b = mat[(2 * i + 1, 2 * j)];
                let c = mat[(2 * i, 2 * j + 1)];
                let d = mat[(2 * i + 1, 2 * j + 1)];
                let ok = f(a, b, c, d);
                mask[(2 * i, 2 * j)] = ok[0];
                mask[(2 * i + 1, 2 * j)] = ok[1];
                mask[(2 * i, 2 * j + 1)] = ok[2];
                mask[(2 * i + 1, 2 * j + 1)] = ok[3];
            }
        }
    }
    mask
}

fn halve_with_thresh(thresh: usize, mat: &DMatrix<u8>) -> Option<DMatrix<u8>> {
    match mat.shape() {
        (r, c) if r * c > thresh => halve(mat),
        _ => None,
    }
}

fn halve(mat: &DMatrix<u8>) -> Option<DMatrix<u8>> {
    multires::halve(mat, |a, b, c, d| {
        let a = a as u16;
        let b = b as u16;
        let c = c as u16;
        let d = d as u16;
        ((a + b + c + d) / 4) as u8
    })
}

fn half_gradient_norm(mat: &DMatrix<u8>) -> DMatrix<u16> {
    multires::halve(mat, |a, b, c, d| {
        let a = a as i32;
        let b = b as i32;
        let c = c as i32;
        let d = d as i32;
        let dx = c + d - a - b;
        let dy = b - a + d - c;
        ((dx * dx + dy * dy) / 8) as u16
    }).unwrap()
}
