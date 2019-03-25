// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Candidates point selection according to
//! "Direct Sparse Odometry", J.Engel, V.Koltun, D. Cremers, PAMI 2018.

use nalgebra::{DMatrix, Scalar};
use num_traits::{self, cast::AsPrimitive, NumCast};
use rand::Rng;
use std::ops::{Add, Div, Mul};

use crate::core::multires;
use crate::misc::helper::div_rem;
use crate::misc::type_aliases::Float;

/// Trait for manipulating numbers types.
pub trait Number<T>:
    Scalar
    + Ord
    + NumCast
    + Add<T, Output = T>
    + Div<T, Output = T>
    + Mul<T, Output = T>
    + AsPrimitive<Float>
    + std::fmt::Display
{
}

impl Number<u16> for u16 {}

/// 0: not picked
/// n: picked at level n
pub type Picked = u8;

/// Configuration of regions.
pub struct RegionConfig<T> {
    /// The region size.
    pub size: usize,
    /// Some coefficients used for the region threshold computation.
    pub threshold_coefs: (Float, T),
}

/// Configuration of blocks.
#[derive(Copy, Clone)]
pub struct BlockConfig {
    /// Base size of a block.
    pub base_size: usize,
    /// Number of levels for picking points in blocks.
    pub nb_levels: usize,
    /// Multiplier factor for block threshold computation.
    pub threshold_factor: Float,
}

/// Configuration of the recursive nature of candidates selection.
/// If the number of points obtained after one iteration is not within
/// given bounds, the algorithm adapts the base block size and re-iterates.
pub struct RecursiveConfig {
    /// Max number of iterations left.
    pub nb_iterations_left: usize,
    /// Low percentage threshold of target number of points.
    pub low_thresh: Float,
    /// High percentage threshold of target number of points.
    pub high_thresh: Float,
    /// Threshold such that if we have random_thresh < points_ratio < high_thresh,
    /// we randomly sample points to have approximatel the desired target number of candidate
    /// points.
    pub random_thresh: Float,
}

/// Default region configuration according to DSO paper and code.
pub const DEFAULT_REGION_CONFIG: RegionConfig<u16> = RegionConfig {
    size: 32,
    threshold_coefs: (1.0, 3), // (2.0, 3) in dso and (1.0, 3) in ldso
};

/// Default block configuration according to DSO paper and code.
pub const DEFAULT_BLOCK_CONFIG: BlockConfig = BlockConfig {
    base_size: 4,
    nb_levels: 3,
    threshold_factor: 0.5,
};

/// Default recursive configuration according to DSO paper and code.
pub const DEFAULT_RECURSIVE_CONFIG: RecursiveConfig = RecursiveConfig {
    nb_iterations_left: 1,
    low_thresh: 0.8,
    high_thresh: 4.0,
    random_thresh: 1.1,
};

/// Select a subset of points satisfying two conditions:
///   * points shall be well-distributed in the image.
///   * higher density where gradients are bigger.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_precision_loss)]
pub fn select<T: Number<T>>(
    gradients: &DMatrix<T>,
    region_config: RegionConfig<T>,
    block_config: BlockConfig,
    recursive_config: RecursiveConfig,
    nb_target: usize,
) -> DMatrix<bool> {
    // Pick all block candidates
    let median_gradients = region_median_gradients(gradients, region_config.size);
    let regions_thresholds = region_thresholds(&median_gradients, region_config.threshold_coefs);
    let (vec_nb_candidates, picked) = pick_all_block_candidates(
        block_config,
        region_config.size,
        &regions_thresholds,
        gradients,
    );
    let nb_candidates: usize = vec_nb_candidates.iter().sum();
    // eprintln!("Number of points picked by level: {:?}", vec_nb_candidates);
    // eprintln!("nb_candidates: {}", nb_candidates);
    let candidates_ratio = nb_candidates as Float / nb_target as Float;
    // The number of selected pixels behave approximately as
    // nb_candidates = K / (block_size + 1)^2 where K is scene dependant.
    // nb_target = K / (target_size + 1)^2
    // So sqrt( candidates_ratio ) = (target_size + 1) / (block_size + 1)
    // and in theory:
    // target_size = sqrt( ratio ) * (block_size + 1) - 1
    let target_size = candidates_ratio.sqrt() * (block_config.base_size as Float + 1.0) - 1.0;
    let target_size = std::cmp::max(1, target_size.round() as i32) as usize;
    // eprintln!("base_size:   {}", block_config.base_size);
    // eprintln!("target_size: {}", target_size);
    if candidates_ratio < recursive_config.low_thresh
        || candidates_ratio > recursive_config.high_thresh
    {
        if target_size != block_config.base_size && recursive_config.nb_iterations_left > 0 {
            let mut b_config = block_config;
            b_config.base_size = target_size;
            let mut rec_config = recursive_config;
            rec_config.nb_iterations_left -= 1;
            select(gradients, region_config, b_config, rec_config, nb_target)
        } else {
            to_mask(&picked)
        }
    } else if candidates_ratio > recursive_config.random_thresh {
        // randomly select a correct % of points
        let mut rng = rand::thread_rng();
        picked.map(|p| p > 0 && rng.gen::<u8>() <= (255.0 / candidates_ratio) as u8)
    } else {
        to_mask(&picked)
    }
}

/// Create a mask of picked points.
fn to_mask(picked: &DMatrix<u8>) -> DMatrix<bool> {
    picked.map(|p| p > 0)
}

/// Pick candidates at all the block levels.
#[allow(clippy::cast_possible_truncation)]
fn pick_all_block_candidates<T: Number<T>>(
    block_config: BlockConfig,
    regions_size: usize,
    regions_thresholds: &DMatrix<T>,
    gradients: &DMatrix<T>,
) -> (Vec<usize>, DMatrix<Picked>) {
    let (nb_rows, nb_cols) = gradients.shape();
    let max_gradients_0 = init_max_gradients(gradients, block_config.base_size);
    let max_gradients_multires =
        multires::limited_sequence(block_config.nb_levels, max_gradients_0, |m| {
            multires::halve(m, max_of_four_gradients)
        });
    let mut threshold_level_coef = 1.0;
    let mut nb_picked = Vec::new();
    let (blocks_rows, blocks_cols) = max_gradients_multires[0].shape();
    let mut mask = DMatrix::repeat(blocks_rows, blocks_cols, true);
    let mut candidates = DMatrix::repeat(nb_rows, nb_cols, 0);
    for (level, max_gradients_level) in max_gradients_multires.iter().enumerate() {
        // call pick_level_block_candidates()
        let (nb_picked_level, mask_next_level, new_candidates) = pick_level_block_candidates(
            threshold_level_coef,
            (level + 1) as u8,
            regions_size,
            regions_thresholds,
            max_gradients_level,
            &mask,
            candidates,
        );
        nb_picked.push(nb_picked_level);
        mask = mask_next_level;
        candidates = new_candidates;
        threshold_level_coef *= block_config.threshold_factor;
    }
    (nb_picked, candidates)
}

/// Retrieve the pixel with max gradient for each block in the image.
fn init_max_gradients<T: Number<T>>(
    gradients: &DMatrix<T>,
    block_size: usize,
) -> DMatrix<(T, usize, usize)> {
    let (nb_rows, nb_cols) = gradients.shape();
    let nb_rows_blocks = match div_rem(nb_rows, block_size) {
        (quot, 0) => quot,
        (quot, _) => quot + 1,
    };
    let nb_cols_blocks = match div_rem(nb_cols, block_size) {
        (quot, 0) => quot,
        (quot, _) => quot + 1,
    };
    DMatrix::from_fn(nb_rows_blocks, nb_cols_blocks, |bi, bj| {
        let start_i = bi * block_size;
        let start_j = bj * block_size;
        let end_i = std::cmp::min(start_i + block_size, nb_rows);
        let end_j = std::cmp::min(start_j + block_size, nb_cols);
        let mut tmp_max = (gradients[(start_i, start_j)], start_i, start_j);
        for j in start_j..end_j {
            for i in start_i..end_i {
                let g = gradients[(i, j)];
                if g > tmp_max.0 {
                    tmp_max = (g, i, j);
                }
            }
        }
        tmp_max
    })
}

/// Retrieve the max and position of 4 gradients.
fn max_of_four_gradients<T: Number<T>>(
    g1: (T, usize, usize),
    g2: (T, usize, usize),
    g3: (T, usize, usize),
    g4: (T, usize, usize),
) -> (T, usize, usize) {
    let g_max = |g_m1: (T, usize, usize), g_m2: (T, usize, usize)| {
        if g_m1.0 < g_m2.0 {
            g_m2
        } else {
            g_m1
        }
    };
    g_max(g1, g_max(g2, g_max(g3, g4)))
}

/// For each block where the mask is "true",
/// Select the pixel with the highest gradient magnitude.
/// Returns the number of selected points and two matrices:
///     * a mask of blocks to test for the next level,
///     * the updated picked candidates.
fn pick_level_block_candidates<T: Number<T>>(
    threshold_level_coef: Float,
    level: Picked,
    regions_size: usize,
    regions_thresholds: &DMatrix<T>,
    max_gradients: &DMatrix<(T, usize, usize)>,
    mask: &DMatrix<bool>,
    candidates: DMatrix<Picked>,
) -> (usize, DMatrix<bool>, DMatrix<Picked>) {
    let (mask_height, mask_width) = mask.shape();
    let mut mask_next_level = DMatrix::repeat(mask_height / 2, mask_width / 2, true);
    let mut candidates = candidates;
    let mut nb_picked = 0;
    // We use mask_width / 2 * 2 to avoid remainder pixels
    for j in 0..(mask_width / 2 * 2) {
        for i in 0..(mask_height / 2 * 2) {
            if mask[(i, j)] {
                let (g2, i_g, j_g) = max_gradients[(i, j)];
                let threshold = regions_thresholds[(i_g / regions_size, j_g / regions_size)];
                if g2.as_() >= threshold_level_coef * threshold.as_() {
                    mask_next_level[(i / 2, j / 2)] = false;
                    candidates[(i_g, j_g)] = level;
                    nb_picked += 1;
                }
            } else {
                mask_next_level[(i / 2, j / 2)] = false;
            }
        }
    }
    (nb_picked, mask_next_level, candidates)
}

/// Smooth the medians and set thresholds given some coefficients (a,b):
/// threshold = a * ( smooth( median ) + b ) ^ 2.
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_wrap)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_possible_truncation)]
fn region_thresholds<T: Number<T>>(median_gradients: &DMatrix<T>, coefs: (Float, T)) -> DMatrix<T> {
    let (nb_rows, nb_cols) = median_gradients.shape();
    DMatrix::from_fn(nb_rows, nb_cols, |i, j| {
        let start_i = std::cmp::max(0, i as i32 - 1) as usize;
        let start_j = std::cmp::max(0, j as i32 - 1) as usize;
        let end_i = std::cmp::min(nb_rows, i + 2);
        let end_j = std::cmp::min(nb_cols, j + 2);
        let mut sum: T = num_traits::cast(0).unwrap();
        let mut nb_elements = 0;
        for j in start_j..end_j {
            for i in start_i..end_i {
                sum = sum + median_gradients[(i, j)];
                nb_elements += 1;
            }
        }
        let (a, b) = coefs;
        let thresh_tmp = sum.as_() / nb_elements as Float + b.as_();
        num_traits::cast(a * thresh_tmp * thresh_tmp).expect("woops")
    })
}

/// Compute median gradients magnitude of each region in the image.
/// The regions on the right and bottom might be smaller.
fn region_median_gradients<T: Number<T>>(gradients: &DMatrix<T>, size: usize) -> DMatrix<T> {
    let (nb_rows, nb_cols) = gradients.shape();
    let nb_rows_regions = match div_rem(nb_rows, size) {
        (quot, 0) => quot,
        (quot, _) => quot + 1,
    };
    let nb_cols_regions = match div_rem(nb_cols, size) {
        (quot, 0) => quot,
        (quot, _) => quot + 1,
    };
    DMatrix::from_fn(nb_rows_regions, nb_cols_regions, |i, j| {
        let height = std::cmp::min(size, nb_rows - i * size);
        let width = std::cmp::min(size, nb_cols - j * size);
        let region_slice = gradients.slice((i * size, j * size), (height, width));
        let mut region_cloned: Vec<T> = region_slice.iter().cloned().collect();
        region_cloned.sort_unstable();
        region_cloned[region_cloned.len() / 2]
    })
}
