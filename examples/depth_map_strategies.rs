extern crate computer_vision_rs as cv;
extern crate image;
extern crate nalgebra as na;

use cv::candidates;
use cv::helper;
use cv::icl_nuim;
use cv::interop;
use cv::inverse_depth;
use cv::multires;

use inverse_depth::InverseDepth;
use na::DMatrix;

// #[allow(dead_code)]
fn main() {
    let nb_img = 1509;
    let accum_eval = (0..nb_img).map(evaluate_icl_image).fold(
        [(0_f32, 0_f32, 0_f32, std::f32::MAX); 2],
        |mut acc, eval| {
            acc.iter_mut()
                .zip(eval.iter())
                .for_each(|(acc_strat, eval_strat)| {
                    acc_strat.0 += eval_strat.0;
                    acc_strat.1 += eval_strat.0 * eval_strat.1.unwrap_or(0.0);
                    acc_strat.2 = acc_strat.2.max(eval_strat.1.unwrap_or(0.0));
                    acc_strat.3 = acc_strat.3.min(eval_strat.1.unwrap_or(0.0));
                });
            acc
        },
    );
    let mean_eval_dso = (
        accum_eval[0].0 / nb_img as f32,
        accum_eval[0].1 / accum_eval[0].0,
        accum_eval[0].2,
        accum_eval[0].3,
    );
    let mean_eval_stat = (
        accum_eval[1].0 / nb_img as f32,
        accum_eval[1].1 / accum_eval[1].0,
        accum_eval[1].2,
        accum_eval[1].3,
    );
    println!("DSO (ratio, rmse, max, min): {:?}", mean_eval_dso);
    println!("Stats (ratio, rmse, max, min): {:?}", mean_eval_stat);
}

// Inverse Depth stuff ###############################################

fn evaluate_icl_image(id: usize) -> [(f32, Option<f32>); 2] {
    let img_path = &format!("icl-rgb/{}.png", id);
    let depth_path = &format!("icl-depth/{}.png", id);
    evaluate_image(img_path, depth_path)
}

fn evaluate_image(rgb_path: &str, depth_path: &str) -> [(f32, Option<f32>); 2] {
    // Load a color image and transform into grayscale.
    let img = image::open(rgb_path).expect("Cannot open image").to_luma();
    let img_matrix = interop::matrix_from_image(img);

    // Read 16 bits PNG image.
    let (width, height, buffer_u16) = helper::read_png_16bits(depth_path).unwrap();
    let depth_map: DMatrix<u16> = DMatrix::from_row_slice(height, width, buffer_u16.as_slice());

    // Evaluate strategies on this image.
    evaluate_all_strategies_for(img_matrix, depth_map)
}

fn evaluate_all_strategies_for(
    img_mat: DMatrix<u8>,
    depth_map: DMatrix<u16>,
) -> [(f32, Option<f32>); 2] {
    // Compute pyramid of matrices.
    let multires_img = multires::mean_pyramid(6, img_mat);

    // Compute pyramid of gradients (without first level).
    let multires_gradient_norm = multires::gradients(&multires_img);

    // canditates
    let multires_candidates = candidates::select(7, &multires_gradient_norm);
    let higher_res_candidate = multires_candidates.last().unwrap();

    // Evaluate strategies.
    let eval_strat =
        |strat: fn(_) -> _| evaluate_strategy_on(&depth_map, higher_res_candidate, strat);
    let dso_eval = eval_strat(inverse_depth::strategy_dso_mean);
    let stat_eval = eval_strat(inverse_depth::strategy_statistically_similar);
    let _random_eval = eval_strat(inverse_depth::strategy_random);
    [dso_eval, stat_eval]
}

fn evaluate_strategy_on<F>(
    depth_map: &DMatrix<u16>,
    sparse_candidates: &DMatrix<bool>,
    strategy: F,
) -> (f32, Option<f32>)
where
    F: Fn(Vec<(f32, f32)>) -> InverseDepth,
{
    // Compute the pyramid of depths maps from ground truth dataset.
    let multires_depth_map = mean_pyramid_u16(6, depth_map.clone());

    // Create a half resolution depth map to fit resolution of candidates map.
    let half_res_depth = &multires_depth_map[1];

    // Transform depth map into an InverseDepth matrix.
    let from_depth = |depth| inverse_depth::from_depth(icl_nuim::DEPTH_SCALE, depth);
    let inverse_depth_mat = half_res_depth.map(from_depth);

    // Only keep InverseDepth values corresponding to point candidates.
    // This is to emulate result of back projection of known points into a new keyframe.
    let inverse_depth_candidates =
        sparse_candidates.zip_map(&inverse_depth_mat, |is_candidate, idepth| {
            if is_candidate {
                idepth
            } else {
                InverseDepth::Unknown
            }
        });

    // Create a multires inverse depth map pyramid
    // with same number of levels than the multires image.
    let fuse = |a, b, c, d| inverse_depth::fuse(a, b, c, d, &strategy);
    let multires_inverse_depth = multires::pyramid_with_max_n_levels(
        5,
        inverse_depth_candidates,
        |mat| mat,
        |mat| multires::halve(&mat, fuse),
    );

    // Compare the lowest resolution inverse depth map of the pyramid
    // with the one from ground truth.
    let lower_res_inverse_depth = multires_inverse_depth.last().unwrap();
    let lower_res_inverse_depth_gt = multires_depth_map.last().unwrap().map(from_depth);
    evaluate_idepth(lower_res_inverse_depth, &lower_res_inverse_depth_gt)
}

// Given an InverseDepth matrix and its ground truth, compute:
//   (1) the ratio of available InverseDepth values (not Unknown or Discarded).
//   (2) the MSE for these available predictions if any.
fn evaluate_idepth(
    idepth_map: &DMatrix<InverseDepth>,
    gt: &DMatrix<InverseDepth>,
) -> (f32, Option<f32>) {
    assert_eq!(idepth_map.shape(), gt.shape());
    let (height, width) = gt.shape();
    let mut count = 0;
    let mut mse = 0.0;
    idepth_map
        .iter()
        .zip(gt.iter())
        .for_each(|(idepth, idepth_gt)| match (idepth, idepth_gt) {
            (InverseDepth::WithVariance(x, _), InverseDepth::WithVariance(x_gt, _)) => {
                count = count + 1;
                mse = mse + (x - x_gt) * (x - x_gt);
            }
            _ => {}
        });
    if count == 0 {
        (0.0, None)
    } else {
        (
            count as f32 / (height as f32 * width as f32),
            Some((mse / count as f32).sqrt()),
        )
    }
}

// Helpers ###########################################################

fn mean_pyramid_u16(max_levels: usize, mat: DMatrix<u16>) -> Vec<DMatrix<u16>> {
    multires::pyramid_with_max_n_levels(
        max_levels,
        mat,
        |m| m,
        |m| {
            multires::halve(m, |a, b, c, d| {
                let a = a as u32;
                let b = b as u32;
                let c = c as u32;
                let d = d as u32;
                ((a + b + c + d) / 4) as u16
            })
        },
    )
}
