extern crate image;
extern crate nalgebra as na;
extern crate rand;

mod helper;
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

    // canditates
    let multires_candidates = candidates(&multires_gradient_norm);

    // Inverse Depth stuff #############

    // Read 16 bits PNG image.
    let (width, height, buffer_u16) = helper::read_png_16bits("icl-depth/0.png").unwrap();

    // Transform it into a matrix.
    let depth_mat: DMatrix<u16> = DMatrix::from_row_slice(height, width, buffer_u16.as_slice());
    let half_res_depth = multires::halve(&depth_mat, |a, b, c, d| {
        ((a as u32 + b as u32 + c as u32 + d as u32) / 4) as u16
    }).unwrap();

    // Transform into an Inverse Depth matrix
    let inverse_depth_mat = half_res_depth.map(inverse_depth);
    let higher_res_candidate = multires_candidates.last().unwrap();
    let inverse_depth_candidates = higher_res_candidate.zip_map(&inverse_depth_mat, |is_candidate, i_depth| {
        if is_candidate { i_depth } else { InverseDepth::Unknown }
    });
    let multires_inverse_depth = multires::pyramid(
        inverse_depth_candidates,
        |mat| mat,
        |mat| multires::halve(&mat, |a,b,c,d| {
            let valid_inverse_depths: Vec<(f32, f32)> = [a,b,c,d].iter().filter_map(inverse_with_variance).collect();
            if valid_inverse_depths.len() == 0 {
                InverseDepth::Unknown
            } else {
                let (idepth, var) = valid_inverse_depths.last().unwrap();
                if rand::random() {
                    InverseDepth::Discarded
                } else {
                    InverseDepth::WithVariance(*idepth, *var)
                }
            }
        })
    );
    multires_inverse_depth
        .iter()
        .map(inverse_depth_visual)
        .map(|mat| interop::image_from_matrix(&mat))
        .enumerate()
        .for_each(|(i,bitmap)| {
            let out_name = &["out/idepth_", i.to_string().as_str(), ".png"].concat();
            bitmap.save(out_name).unwrap();
        });
}

// Inverse Depth stuff ###############################################

#[derive(Copy, Clone, PartialEq, Debug)]
enum InverseDepth {
    Unknown,
    Discarded,
    WithVariance(f32, f32),
}

fn inverse_with_variance(inverse_depth: &InverseDepth) -> Option<(f32, f32)> {
    if let InverseDepth::WithVariance(idepth, var) = inverse_depth {
        Some((*idepth, *var))
    } else {
        None
    }
}

fn inverse_depth_visual(inverse_mat: &DMatrix<InverseDepth>) -> DMatrix<u8> {
    inverse_mat.map(|idepth| {
        match idepth {
            InverseDepth::Unknown => 0u8,
            InverseDepth::Discarded => 50u8,
            InverseDepth::WithVariance(_,_) => 255u8,
        }
    })
}

// Transform depth value from dataset into an inverse depth value.
// STD is assimilated to 10cm so variance = ( 1 / 0.10 ) ^ 2 = 100.
fn inverse_depth(depth: u16) -> InverseDepth {
    match depth {
        0 => InverseDepth::Unknown,
        _ => InverseDepth::WithVariance(5000f32 / depth as f32, 100f32),
    }
}

// Multires Pyramid stuff ############################################

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

// Gradients stuff ###################################################

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

// Candidates stuff ##################################################

fn candidates(gradients: &Vec<DMatrix<u16>>) -> Vec<DMatrix<bool>> {
    let (nrows, ncols) = gradients.last().unwrap().shape();
    let pre_mask = DMatrix::repeat(nrows, ncols, true);
    let mut init_candidates = Vec::new();
    init_candidates.push(pre_mask);
    let multires_candidates = gradients
        .iter()
        .rev() // start with lower res
        .skip(1) // skip lower since all points are good
        .fold(init_candidates, |mut multires_masks, grad_mat| {
            let new_mask = bloc_2x2_filter(&multires_masks.last().unwrap(), &grad_mat, higher_than_mean_with_thresh);
            multires_masks.push(new_mask);
            multires_masks
        });
    multires_candidates
}

fn higher_than_mean_with_thresh(a: u16, b: u16, c: u16, d: u16) -> [bool; 4] {
    let thresh = 7;
    let mut temp = [(a, 0usize), (b, 1usize), (c, 2usize), (d, 3usize)];
    temp.sort_unstable_by(|(x, _), (y, _)| x.cmp(y));
    let (_, first) = temp[3];
    let (x, second) = temp[2];
    let (y, _) = temp[1];
    let mut result = [false; 4];
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
