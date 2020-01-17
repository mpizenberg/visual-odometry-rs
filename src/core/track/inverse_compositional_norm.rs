// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Types and functions to implement an inverse compositional tracking algorithm.
//!
//! Implementation of "Lucas-kanade 20 years on: A unifying framework"
//! in the inverse compositional case.
//! The warping function is parameterized by the Lie Algebra of twists se(3).

use itertools::izip;
use nalgebra::DMatrix;

use crate::core::{
    camera::{Camera, Intrinsics},
    candidates::coarse_to_fine as candidates,
    gradient,
    inverse_depth::{self, InverseDepth},
    multires,
    track::lm_optimizer_norm::{self as lm_optimizer, LMOptimizerState},
};
use crate::math::optimizer::State as _;
use crate::misc::helper;
use crate::misc::type_aliases::{Float, Iso3, Mat6, Point2, Point3, Vec6};

/// Type alias to easily spot vectors that are indexed over multi-resolution levels.
pub type Levels<T> = Vec<T>;

/// Struct used for tracking the camera at each frame.
/// Can only be constructed by initialization from a `Config`.
#[derive(Debug, Clone)]
pub struct Tracker {
    config: Config,
    state: State,
}

/// Configuration of the Tracker.
#[derive(Debug, Clone)]
pub struct Config {
    /// Number of levels in the multi-resolution pyramids of images.
    pub nb_levels: usize,
    /// Threshold for the candidates selection algorithm.
    pub candidates_diff_threshold: u16,
    /// Scale of the depth 16 bit images.
    /// This is 5000.0 for the TUM RGB-D dataset.
    pub depth_scale: Float,
    /// Camera intrinsic parameters.
    pub intrinsics: Intrinsics,
    /// Default variance of the inverse depth values coming from the depth map.
    pub idepth_variance: Float,
}

/// Internal state of the tracker.
#[derive(Debug, Clone)]
struct State {
    keyframe_multires_data: MultiresData,
    keyframe_depth_timestamp: f64,
    keyframe_img_timestamp: f64,
    keyframe_pose: Iso3,
    current_frame_depth_timestamp: f64,
    current_frame_img_timestamp: f64,
    current_frame_pose: Iso3,
}

/// Mostly multi-resolution data related to the frame.
#[derive(Debug, Clone)]
#[allow(clippy::type_complexity)]
struct MultiresData {
    intrinsics_multires: Levels<Intrinsics>,
    img_multires: Levels<DMatrix<u8>>,
    usable_candidates_multires: Levels<(Vec<(usize, usize)>, Vec<Float>)>,
    jacobians_multires: Levels<Vec<Vec6>>,
    hessians_multires: Levels<Vec<Mat6>>,
}

impl Config {
    /// Initialize a tracker with the first RGB-D frame.
    pub fn init(
        self,
        keyframe_depth_timestamp: f64,
        depth_map: &DMatrix<u16>,
        keyframe_img_timestamp: f64,
        img: DMatrix<u8>,
    ) -> Tracker {
        // Precompute multi-resolution first frame data.
        let intrinsics_multires = self.intrinsics.clone().multi_res(self.nb_levels);
        let img_multires = multires::mean_pyramid(self.nb_levels, img);
        let keyframe_multires_data =
            precompute_multires_data(&self, depth_map, intrinsics_multires, img_multires);

        // Regroup everything under the returned Tracker.
        Tracker {
            state: State {
                keyframe_multires_data,
                keyframe_depth_timestamp,
                keyframe_img_timestamp,
                keyframe_pose: Iso3::identity(),
                current_frame_depth_timestamp: keyframe_depth_timestamp,
                current_frame_img_timestamp: keyframe_img_timestamp,
                current_frame_pose: Iso3::identity(),
            },
            config: self,
        }
    }
} // impl Config

/// Precompute the multi-resolution data of a frame.
#[allow(clippy::used_underscore_binding)]
fn precompute_multires_data(
    config: &Config,
    depth_map: &DMatrix<u16>,
    intrinsics_multires: Levels<Intrinsics>,
    img_multires: Levels<DMatrix<u8>>,
) -> MultiresData {
    // Precompute multi-resolution of keyframe gradients.
    // let mut gradients_multires = multires::gradients_xy(&img_multires);
    let mut gradients_multires = multires::gradients_xy_smooth(&img_multires);
    gradients_multires.insert(0, gradient::centered(&img_multires[0]));
    let gradients_squared_norm_multires: Vec<_> = gradients_multires
        .iter()
        .map(|(gx, gy)| gradient::squared_norm(gx, gy))
        .collect();

    // Precompute mask of candidate points for tracking.
    let candidates_points = candidates::select(
        config.candidates_diff_threshold,
        &gradients_squared_norm_multires,
    )
    .pop()
    .unwrap();

    // Only keep the "usable" points, i.e. those with a known depth information.
    let from_depth = |z| inverse_depth::from_depth(config.depth_scale, z, config.idepth_variance);
    let idepth_candidates = helper::zip_mask_map(
        depth_map,
        &candidates_points,
        InverseDepth::Unknown,
        from_depth,
    );
    let fuse = |a, b, c, d| inverse_depth::fuse(a, b, c, d, inverse_depth::strategy_dso_mean);
    let idepth_multires = multires::limited_sequence(config.nb_levels, idepth_candidates, |m| {
        multires::halve(m, fuse)
    });
    let usable_candidates_multires: Levels<_> = idepth_multires.iter().map(extract_z).collect();

    // Precompute the Jacobians.
    let jacobians_multires: Levels<Vec<Vec6>> = izip!(
        &intrinsics_multires,
        &usable_candidates_multires,
        &gradients_multires,
    )
    // .map(|(intrinsics, (coord, _z), (gx, gy))| warp_jacobians(intrinsics, coord, _z, gx, gy))
    .map(|(intrinsics, (coord, _z), (gx, gy))| {
        warp_jacobians_normalized(intrinsics, coord, _z, gx, gy)
    })
    .collect();

    // Precompute the Hessians.
    let hessians_multires: Levels<_> = jacobians_multires.iter().map(hessians_vec).collect();

    // Regroup everything under a MultiresData.
    MultiresData {
        intrinsics_multires,
        img_multires,
        usable_candidates_multires,
        jacobians_multires,
        hessians_multires,
    }
}

impl Tracker {
    /// Track a new frame.
    /// Internally mutates the tracker state.
    ///
    /// You can use `tracker.current_frame()` after tracking to retrieve the new frame pose.
    #[allow(clippy::used_underscore_binding)]
    #[allow(clippy::cast_precision_loss)]
    pub fn track(
        &mut self,
        force_keyframe: bool,
        depth_time: f64,
        depth_map: &DMatrix<u16>,
        img_time: f64,
        img: DMatrix<u8>,
    ) -> bool {
        let mut lm_model = self.state.current_frame_pose.inverse() * self.state.keyframe_pose;
        let img_multires = multires::mean_pyramid(self.config.nb_levels, img);
        let keyframe_data = &self.state.keyframe_multires_data;
        let mut optimization_went_well = true;
        for lvl in (0..self.config.nb_levels).rev() {
            // eprintln!("--- Level {}", lvl);
            let obs = lm_optimizer::Obs {
                intrinsics: &keyframe_data.intrinsics_multires[lvl],
                template: &keyframe_data.img_multires[lvl],
                image: &img_multires[lvl],
                coordinates: &keyframe_data.usable_candidates_multires[lvl].0,
                _z_candidates: &keyframe_data.usable_candidates_multires[lvl].1,
                jacobians: &keyframe_data.jacobians_multires[lvl],
                hessians: &keyframe_data.hessians_multires[lvl],
            };
            match LMOptimizerState::iterative_solve(&obs, lm_model) {
                Ok((lm_state, _)) => {
                    lm_model = lm_state.eval_data.model;
                }
                Err(err) => {
                    eprintln!("{}", err);
                    optimization_went_well = false;
                    break;
                }
            }
        }

        // Update current frame info in tracker.
        self.state.current_frame_depth_timestamp = depth_time;
        self.state.current_frame_img_timestamp = img_time;
        if optimization_went_well {
            self.state.current_frame_pose = self.state.keyframe_pose * lm_model.inverse();
        }

        // Check if we need to change the keyframe.
        let (coordinates, _z_candidates) = keyframe_data.usable_candidates_multires.last().unwrap();
        let intrinsics = keyframe_data.intrinsics_multires.last().unwrap();
        let optical_flow_sum: Float = _z_candidates
            .iter()
            .zip(coordinates.iter())
            .map(|(&_z, &(x, y))| {
                let (u, v) = warp(&lm_model, x as Float, y as Float, _z, intrinsics);
                (x as Float - u).abs() + (y as Float - v).abs()
            })
            .sum();
        let optical_flow = optical_flow_sum / _z_candidates.len() as Float;
        eprintln!("Optical_flow: {}", optical_flow);

        let change_keyframe = force_keyframe || optical_flow >= 2.0;

        // In case of keyframe change, update all keyframe info with current frame.
        if change_keyframe {
            let delta_time = depth_time - self.state.keyframe_depth_timestamp;
            eprintln!("Changing keyframe after: {} seconds", delta_time);
            self.state.keyframe_multires_data = precompute_multires_data(
                &self.config,
                depth_map,
                keyframe_data.intrinsics_multires.clone(),
                img_multires,
            );
            self.state.keyframe_depth_timestamp = depth_time;
            self.state.keyframe_img_timestamp = img_time;
            self.state.keyframe_pose = self.state.current_frame_pose;
        }

        change_keyframe
    } // track

    /// Retrieve the current frame timestamp (of depth image) and pose.
    pub fn current_frame(&self) -> (f64, Iso3) {
        (
            self.state.current_frame_depth_timestamp,
            self.state.current_frame_pose,
        )
    }

    /// Collect 3d points of keyframe.
    pub fn points_3d(&self) -> Vec<Point3> {
        let intrinsics = &self.config.intrinsics;
        let extrinsics = &self.state.keyframe_pose;
        let camera = Camera::new(intrinsics.clone(), extrinsics.clone());
        let (coordinates, _z) = &self.state.keyframe_multires_data.usable_candidates_multires[0];
        coordinates
            .iter()
            .zip(_z.iter())
            .map(|(&(x, y), &_z)| camera.back_project(Point2::new(x as f32, y as f32), 1.0 / _z))
            .collect()
    }

    /// Retrieve current candidate coordinates at half resolution.
    pub fn keyframe_candidates(&self) -> &[(usize, usize)] {
        &self.state.keyframe_multires_data.usable_candidates_multires[1].0[..]
    }

    /// Retrieve current candidate inverse depths at half resolution.
    pub fn keyframe_candidates_idepths(&self) -> &[Float] {
        &self.state.keyframe_multires_data.usable_candidates_multires[1].1[..]
    }

    /// Retrieve current keyframe at half resolution.
    pub fn keyframe_img(&self) -> &DMatrix<u8> {
        &self.state.keyframe_multires_data.img_multires[1]
    }

    /// Retrieve camera intrinsics.
    pub fn intrinsics(&self) -> &Levels<Intrinsics> {
        &self.state.keyframe_multires_data.intrinsics_multires
    }

    /// Retrieve tracking config.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Reset camera pose.
    pub fn reset_pose(&mut self, kf_pose: Iso3, current_pose: Iso3) {
        self.state.keyframe_pose = kf_pose;
        self.state.current_frame_pose = current_pose;
    }

    /// Compute the reprojection error of a given camera pose for a given image.
    /// Return (inside_ratio, mean_error).
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::used_underscore_binding)]
    pub fn reprojection_error(&self, pose: &Iso3, image_full_res: &DMatrix<u8>) -> (Float, Float) {
        let mut count = 0;
        let mut sum_error = 0.0;
        let level = 4;
        let (coords, idepths) =
            &self.state.keyframe_multires_data.usable_candidates_multires[level];
        let template = &self.state.keyframe_multires_data.img_multires[level];
        let img_multires = multires::mean_pyramid(self.config.nb_levels, image_full_res.clone());
        let image = &img_multires[level];
        let intrinsics = &self.state.keyframe_multires_data.intrinsics_multires[level];
        let projection = pose.inverse() * self.state.keyframe_pose;
        for (&(x, y), &_z) in coords.iter().zip(idepths.iter()) {
            // check if warp(x,y) is inside the image
            let (u, v) = warp(&projection, x as Float, y as Float, _z, intrinsics);
            if let Some(im) = interpolate(u, v, image) {
                count += 1;
                let tmp = template[(y, x)];
                let residual = im - Float::from(tmp);
                sum_error += residual.abs();
            }
        }
        let inside_ratio = count as Float / idepths.len() as Float;
        let mean_error = sum_error / count as Float;
        (inside_ratio, mean_error)
    }
} // impl Tracker

// Helper ######################################################################

// fn angle(uq: UnitQuaternion<Float>) -> Float {
//     let w = uq.into_inner().scalar();
//     2.0 * uq.into_inner().vector().norm().atan2(w)
// }

/// Extract known inverse depth values (and coordinates) into vectorized data.
#[allow(clippy::used_underscore_binding)]
fn extract_z(idepth_mat: &DMatrix<InverseDepth>) -> (Vec<(usize, usize)>, Vec<Float>) {
    let mut u = 0;
    let mut v = 0;
    // TODO: can allocating with a known max size improve performances?
    let mut coordinates = Vec::new();
    let mut _z_vec = Vec::new();
    let (nb_rows, _) = idepth_mat.shape();
    for idepth in idepth_mat.iter() {
        if let InverseDepth::WithVariance(_z, _) = *idepth {
            coordinates.push((u, v));
            _z_vec.push(_z);
        }
        v += 1;
        if v >= nb_rows {
            u += 1;
            v = 0;
        }
    }
    (coordinates, _z_vec)
}

/// Precompute jacobians for each candidate with normalized coordinates.
#[allow(clippy::used_underscore_binding)]
#[allow(clippy::cast_precision_loss)]
fn warp_jacobians_normalized(
    intrinsics: &Intrinsics,
    coordinates: &[(usize, usize)],
    _z_candidates: &[Float],
    grad_x: &DMatrix<i16>,
    grad_y: &DMatrix<i16>,
) -> Vec<Vec6> {
    let (fu, fv) = intrinsics.focal;
    // Iterate on inverse depth candidates
    coordinates
        .iter()
        .zip(_z_candidates.iter())
        .map(|(&(u, v), &_z)| {
            let gu = fu * Float::from(grad_x[(v, u)]);
            let gv = fv * Float::from(grad_y[(v, u)]);
            let normalized = intrinsics.back_project(Point2::new(u as f32, v as f32), 1.0 / _z);
            warp_jacobian_normalized_at(gu, gv, normalized)
        })
        .collect()
}

/// Precompute jacobians for each candidate.
#[allow(clippy::used_underscore_binding)]
#[allow(clippy::cast_precision_loss)]
fn warp_jacobians(
    intrinsics: &Intrinsics,
    coordinates: &[(usize, usize)],
    _z_candidates: &[Float],
    grad_x: &DMatrix<i16>,
    grad_y: &DMatrix<i16>,
) -> Vec<Vec6> {
    // Bind intrinsics to shorter names
    let (cu, cv) = intrinsics.principal_point;
    let (fu, fv) = intrinsics.focal;
    let s = intrinsics.skew;

    // Iterate on inverse depth candidates
    coordinates
        .iter()
        .zip(_z_candidates.iter())
        .map(|(&(u, v), &_z)| {
            let gu = Float::from(grad_x[(v, u)]);
            let gv = Float::from(grad_y[(v, u)]);
            warp_jacobian_at(gu, gv, u as Float, v as Float, _z, cu, cv, fu, fv, s)
        })
        .collect()
}

/// Jacobian of the warping function for the inverse compositional algorithm
/// with normalized coordinates.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::many_single_char_names)]
#[allow(clippy::used_underscore_binding)]
#[allow(clippy::similar_names)]
fn warp_jacobian_normalized_at(gu: Float, gv: Float, normalized: Point3) -> Vec6 {
    // Intermediate computations
    let x = normalized.x;
    let y = normalized.y;
    let _z = 1.0 / normalized.z;
    let x_z = x * _z;
    let y_z = y * _z;
    let xy_zz = x_z * y_z;

    // Jacobian of the warp
    Vec6::new(
        gu * _z,                                //
        gv * _z,                                //  linear velocity terms
        -_z * (x_z * gu + y_z * gv),            //  ___
        -(gu * xy_zz + gv * (1.0 + y_z * y_z)), //
        gu * (1.0 + x_z * x_z) + gv * xy_zz,    //  angular velocity terms
        gv * x_z - gu * y_z,                    //
    )
}

/// Jacobian of the warping function for the inverse compositional algorithm.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::many_single_char_names)]
#[allow(clippy::used_underscore_binding)]
#[allow(clippy::similar_names)]
fn warp_jacobian_at(
    gu: Float,
    gv: Float,
    u: Float,
    v: Float,
    _z: Float,
    cu: Float,
    cv: Float,
    fu: Float,
    fv: Float,
    s: Float,
) -> Vec6 {
    // Intermediate computations
    let a = u - cu;
    let b = v - cv;
    let c = a * fv - s * b;
    let _fv = 1.0 / fv;
    let _fuv = 1.0 / (fu * fv);

    // Jacobian of the warp
    Vec6::new(
        gu * _z * fu,                                       //
        _z * (gu * s + gv * fv),                            //  linear velocity terms
        -_z * (gu * a + gv * b),                            //  ___
        gu * (-a * b * _fv - s) + gv * (-b * b * _fv - fv), //
        gu * (a * c * _fuv + fu) + gv * (b * c * _fuv),     //  angular velocity terms
        gu * (-fu * fu * b + s * c) * _fuv + gv * (c / fu), //
    )
}

/// Compute hessians components for each candidate point.
#[allow(clippy::ptr_arg)] // TODO: Applying clippy lint here results in compilation error.
fn hessians_vec(jacobians: &Vec<Vec6>) -> Vec<Mat6> {
    // TODO: might be better to inline this within the function computing the jacobians.
    jacobians.iter().map(|j| j * j.transpose()).collect()
}

/// Warp a point from an image to another by a given rigid body motion.
#[allow(clippy::used_underscore_binding)]
fn warp(model: &Iso3, x: Float, y: Float, _z: Float, intrinsics: &Intrinsics) -> (Float, Float) {
    // TODO: maybe move into the camera module?
    let x1 = intrinsics.back_project(Point2::new(x, y), 1.0 / _z);
    let x2 = model * x1;
    let uvz2 = intrinsics.project(x2);
    (uvz2.x / uvz2.z, uvz2.y / uvz2.z)
}

/// Simple linear interpolation of a pixel with floating point coordinates.
/// Return `None` if the point is outside of the image boundaries.
/// TODO: duplicate of function in lm_optimizer.rs.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_precision_loss)]
fn interpolate(x: Float, y: Float, image: &DMatrix<u8>) -> Option<Float> {
    let (height, width) = image.shape();
    let u = x.floor();
    let v = y.floor();
    if u >= 0.0 && u < (width - 2) as Float && v >= 0.0 && v < (height - 2) as Float {
        let u_0 = u as usize;
        let v_0 = v as usize;
        let u_1 = u_0 + 1;
        let v_1 = v_0 + 1;
        let vu_00 = Float::from(image[(v_0, u_0)]);
        let vu_10 = Float::from(image[(v_1, u_0)]);
        let vu_01 = Float::from(image[(v_0, u_1)]);
        let vu_11 = Float::from(image[(v_1, u_1)]);
        let a = x - u;
        let b = y - v;
        Some(
            (1.0 - b) * (1.0 - a) * vu_00
                + b * (1.0 - a) * vu_10
                + (1.0 - b) * a * vu_01
                + b * a * vu_11,
        )
    } else {
        None
    }
}
