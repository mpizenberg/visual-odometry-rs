use itertools::izip;
use nalgebra::DMatrix;

use crate::core::{
    camera::Intrinsics,
    candidates,
    inverse_depth::{self, InverseDepth},
    multires,
};
use crate::math::optimizer::Optimizer;
use crate::misc::helper;
use crate::misc::type_aliases::{Float, Iso3, Mat6, Point2, Vec6};

pub mod lm_optimizer;

pub type Levels<T> = Vec<T>;

pub struct Tracker {
    config: Config,
    state: State,
}

pub struct Config {
    pub nb_levels: usize,
    pub candidates_diff_threshold: u16,
    pub depth_scale: Float,
    pub intrinsics: Intrinsics,
}

pub struct State {
    keyframe_multires_data: MultiresData,
    keyframe_depth_timestamp: f64,
    keyframe_img_timestamp: f64,
    keyframe_pose: Iso3,
    current_frame_depth_timestamp: f64,
    current_frame_img_timestamp: f64,
    current_frame_pose: Iso3,
}

struct MultiresData {
    intrinsics_multires: Levels<Intrinsics>,
    img_multires: Levels<DMatrix<u8>>,
    usable_candidates_multires: Levels<(Vec<(usize, usize)>, Vec<Float>)>,
    jacobians_multires: Levels<Vec<Vec6>>,
    hessians_multires: Levels<Vec<Mat6>>,
}

impl Config {
    pub fn init(
        self,
        keyframe_depth_timestamp: f64,
        depth_map: DMatrix<u16>,
        keyframe_img_timestamp: f64,
        img: DMatrix<u8>,
    ) -> Tracker {
        // Precompute multi-resolution first frame data.
        let intrinsics_multires = self.intrinsics.clone().multi_res(self.nb_levels);
        let img_multires = multires::mean_pyramid(self.nb_levels, img);
        let keyframe_multires_data =
            precompute_multires_data(&self, &depth_map, intrinsics_multires, img_multires);

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

fn precompute_multires_data(
    config: &Config,
    depth_map: &DMatrix<u16>,
    intrinsics_multires: Levels<Intrinsics>,
    img_multires: Levels<DMatrix<u8>>,
) -> MultiresData {
    // Precompute multi-resolution of keyframe gradients.
    let mut gradients_multires = multires::gradients_xy(&img_multires);
    gradients_multires.insert(0, im_gradient(&img_multires[0]));
    let gradients_squared_norm_multires: Vec<_> = gradients_multires
        .iter()
        .map(|(gx, gy)| grad_squared_norm(gx, gy))
        .collect();

    // Precompute mask of candidate points for tracking.
    let candidates_points = candidates::select(
        config.candidates_diff_threshold,
        &gradients_squared_norm_multires,
    )
    .pop()
    .unwrap();

    // Only keep the "usable" points, i.e. those with a known depth information.
    let from_depth = |z| inverse_depth::from_depth(config.depth_scale, z);
    let idepth_candidates = helper::zip_mask_map(
        &depth_map,
        &candidates_points,
        InverseDepth::Unknown,
        from_depth,
    );
    let fuse = |a, b, c, d| inverse_depth::fuse(a, b, c, d, inverse_depth::strategy_dso_mean);
    let idepth_multires = multires::limited_sequence(
        config.nb_levels,
        idepth_candidates,
        |m| m,
        |m| multires::halve(&m, fuse),
    );
    let usable_candidates_multires: Levels<_> = idepth_multires.iter().map(extract_z).collect();

    // Precompute the Jacobians.
    let jacobians_multires: Levels<Vec<Vec6>> = izip!(
        &intrinsics_multires,
        &usable_candidates_multires,
        &gradients_multires,
    )
    .map(|(intrinsics, (coord, _z), (gx, gy))| warp_jacobians(intrinsics, coord, _z, gx, gy))
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
    pub fn track(
        &mut self,
        depth_time: f64,
        depth_map: DMatrix<u16>,
        img_time: f64,
        img: DMatrix<u8>,
    ) {
        let mut lm_model = self.state.current_frame_pose.inverse() * self.state.keyframe_pose;
        let img_multires = multires::mean_pyramid(self.config.nb_levels, img);
        let keyframe_data = &self.state.keyframe_multires_data;
        let mut optimization_went_well = true;
        for lvl in (0..self.config.nb_levels).rev() {
            let obs = lm_optimizer::Obs {
                intrinsics: &keyframe_data.intrinsics_multires[lvl],
                template: &keyframe_data.img_multires[lvl],
                image: &img_multires[lvl],
                coordinates: &keyframe_data.usable_candidates_multires[lvl].0,
                _z_candidates: &keyframe_data.usable_candidates_multires[lvl].1,
                jacobians: &keyframe_data.jacobians_multires[lvl],
                hessians: &keyframe_data.hessians_multires[lvl],
            };
            let data = lm_optimizer::LMOptimizer::init(&obs, lm_model).unwrap();
            let lm_state = lm_optimizer::LMState { lm_coef: 0.1, data };
            match lm_optimizer::LMOptimizer::iterative(&obs, lm_state) {
                Some((lm_state, _)) => {
                    lm_model = lm_state.data.model;
                }
                None => {
                    eprintln!("Iterations did not converge!");
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

        let change_keyframe = optical_flow >= 1.0;

        // In case of keyframe change, update all keyframe info with current frame.
        if change_keyframe {
            let delta_time = depth_time - self.state.keyframe_depth_timestamp;
            eprintln!("Changing keyframe after: {} seconds", delta_time);
            self.state.keyframe_multires_data = precompute_multires_data(
                &self.config,
                &depth_map,
                keyframe_data.intrinsics_multires.clone(),
                img_multires,
            );
            self.state.keyframe_depth_timestamp = depth_time;
            self.state.keyframe_img_timestamp = img_time;
            self.state.keyframe_pose = self.state.current_frame_pose;
        }
    }

    pub fn current_frame(&self) -> (f64, Iso3) {
        (
            self.state.current_frame_depth_timestamp,
            self.state.current_frame_pose,
        )
    }
} // impl Tracker

// Helper ######################################################################

// fn angle(uq: UnitQuaternion<Float>) -> Float {
//     let w = uq.into_inner().scalar();
//     2.0 * uq.into_inner().vector().norm().atan2(w)
// }

fn im_gradient(im: &DMatrix<u8>) -> (DMatrix<i16>, DMatrix<i16>) {
    let (nb_rows, nb_cols) = im.shape();
    let top = im.slice((0, 1), (nb_rows - 2, nb_cols - 2));
    let bottom = im.slice((2, 1), (nb_rows - 2, nb_cols - 2));
    let left = im.slice((1, 0), (nb_rows - 2, nb_cols - 2));
    let right = im.slice((1, 2), (nb_rows - 2, nb_cols - 2));
    let mut grad_x = DMatrix::zeros(nb_rows, nb_cols);
    let mut grad_y = DMatrix::zeros(nb_rows, nb_cols);
    let mut grad_x_inner = grad_x.slice_mut((1, 1), (nb_rows - 2, nb_cols - 2));
    let mut grad_y_inner = grad_y.slice_mut((1, 1), (nb_rows - 2, nb_cols - 2));
    for j in 0..nb_cols - 2 {
        for i in 0..nb_rows - 2 {
            grad_x_inner[(i, j)] = (right[(i, j)] as i16 - left[(i, j)] as i16) / 2;
            grad_y_inner[(i, j)] = (bottom[(i, j)] as i16 - top[(i, j)] as i16) / 2;
        }
    }
    (grad_x, grad_y)
}

fn grad_squared_norm(grad_x: &DMatrix<i16>, grad_y: &DMatrix<i16>) -> DMatrix<u16> {
    grad_x.zip_map(grad_y, |gx, gy| {
        let gx = gx as i32;
        let gy = gy as i32;
        (gx * gx + gy * gy) as u16
    })
}

fn extract_z(idepth_mat: &DMatrix<InverseDepth>) -> (Vec<(usize, usize)>, Vec<Float>) {
    let mut u = 0;
    let mut v = 0;
    let mut coordinates = Vec::new();
    let mut _z_vec = Vec::new();
    let (nb_rows, _) = idepth_mat.shape();
    for idepth in idepth_mat.iter() {
        if let &InverseDepth::WithVariance(_z, _) = idepth {
            coordinates.push((u, v));
            _z_vec.push(_z);
        }
        v = v + 1;
        if v >= nb_rows {
            u = u + 1;
            v = 0;
        }
    }
    (coordinates, _z_vec)
}

fn warp_jacobians(
    intrinsics: &Intrinsics,
    coordinates: &Vec<(usize, usize)>,
    _z_candidates: &Vec<Float>,
    grad_x: &DMatrix<i16>,
    grad_y: &DMatrix<i16>,
) -> Vec<Vec6> {
    // Bind intrinsics to shorter names
    let (cu, cv) = intrinsics.principal_point;
    let (su, sv) = intrinsics.scaling;
    let fu = su * intrinsics.focal_length;
    let fv = sv * intrinsics.focal_length;
    let s = intrinsics.skew;

    // Iterate on inverse depth candidates
    coordinates
        .iter()
        .zip(_z_candidates.iter())
        .map(|(&(u, v), &_z)| {
            let gu = grad_x[(v, u)] as Float;
            let gv = grad_y[(v, u)] as Float;
            warp_jacobian_at(gu, gv, u as Float, v as Float, _z, cu, cv, fu, fv, s)
        })
        .collect()
}

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
    #[rustfmt::skip]
    let jac = Vec6::new(
        gu * _z * fu,                                        //
        _z * (gu * s + gv * fv),                             //  linear velocity terms
        -_z * (gu * a + gv * b),                             //  ___
        gu * (-a * b * _fv - s) + gv * (-b * b * _fv - fv),  //
        gu * (a * c * _fuv + fu) + gv * (b * c * _fuv),      //  angular velocity terms
        gu * (-fu * fu * b + s * c) * _fuv + gv * (c / fu),  //
    );
    jac
}

fn hessians_vec(jacobians: &Vec<Vec6>) -> Vec<Mat6> {
    jacobians.iter().map(|j| j * j.transpose()).collect()
}

fn warp(model: &Iso3, x: Float, y: Float, _z: Float, intrinsics: &Intrinsics) -> (Float, Float) {
    let x1 = intrinsics.back_project(Point2::new(x, y), 1.0 / _z);
    let x2 = model * x1;
    let uvz2 = intrinsics.project(x2);
    (uvz2.x / uvz2.z, uvz2.y / uvz2.z)
}
