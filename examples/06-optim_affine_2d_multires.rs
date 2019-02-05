extern crate computer_vision_rs as cv;
extern crate itertools;
extern crate nalgebra as na;

use cv::interop;
use cv::multires;
use cv::optimization_bis::{Continue, Optimizer, State};
use na::{geometry::Affine2, DMatrix, Matrix3, Matrix6, Vector6};
use std::f32;

fn main() {
    let _ = run();
}

fn run() -> Result<(), f32> {
    // Read the template image
    let tmp = image::open("data/alignment/template.png")
        .unwrap()
        .to_luma();
    let tmp = interop::matrix_from_image(tmp);
    let tmp_multires = multires::mean_pyramid(5, tmp);

    // Read the transformed image
    let img = image::open("data/alignment/image.png").unwrap().to_luma();
    let img = interop::matrix_from_image(img);
    let img_multires = multires::mean_pyramid(5, img);

    // Precompute template gradients
    let grad_multires = multires::gradients_xy(&tmp_multires);

    // Precompute the Jacobians (cf baker unify)
    let jacobians_multires: Vec<_> = grad_multires
        .iter()
        .map(|(gx, gy)| affine_jacobians(gx, gy))
        .collect();

    // Precompute the Hessians
    let hessians_multires: Vec<_> = jacobians_multires.iter().map(hessians_vec).collect();

    // Multi-resolution optimization
    let mut model = Vec6::zeros();
    for lvl in (0..grad_multires.len()).rev() {
        println!("---------------- Level {}:", lvl + 1);
        model.a = 2.0 * model.a;
        model.b = 2.0 * model.b;
        let obs = Obs {
            template: &tmp_multires[lvl + 1],
            image: &img_multires[lvl + 1],
            jacobians: &jacobians_multires[lvl],
            hessians: &hessians_multires[lvl],
        };
        let data = LMOptimizer::init(&obs, model)?;
        let state = LMState { lm_coef: 0.1, data };
        let (state, _) = LMOptimizer::iterative(&obs, state);
        model = state.data.model;
    }

    // Final (full-res) optimization
    println!("---------------- Level {}:", 0);
    let template = &tmp_multires[0];
    let image = &img_multires[0];
    let (grad_x, grad_y) = gradient(template);
    let jacobians = affine_jacobians(&grad_x, &grad_y);
    let hessians = hessians_vec(&jacobians);
    let obs = Obs {
        template,
        image,
        jacobians: &jacobians,
        hessians: &hessians,
    };
    model.a = 2.0 * model.a;
    model.b = 2.0 * model.b;
    let data = LMOptimizer::init(&obs, model)?;
    let state = LMState { lm_coef: 0.1, data };
    let (state, _) = LMOptimizer::iterative(&obs, state);
    model = state.data.model;

    println!("Final model: {}", warp_mat(model));
    Ok(())
}

struct LMOptimizer;
struct LMState {
    lm_coef: f32,
    data: LMData,
}
struct LMData {
    hessian: Mat6,
    gradient: Vec6,
    energy: f32,
    model: Vec6,
}
type LMPartialState = Result<LMData, f32>;
type PreEval = (Vec<usize>, Vec<f32>, f32);
struct Obs<'a> {
    template: &'a DMatrix<u8>,
    image: &'a DMatrix<u8>,
    jacobians: &'a Vec<Vec6>,
    hessians: &'a Vec<Mat6>,
}
type Vec6 = Vector6<f32>;
type Mat6 = Matrix6<f32>;
type Mat3 = Matrix3<f32>;

impl State<Vec6, f32> for LMState {
    fn model(&self) -> &Vec6 {
        &self.data.model
    }
    fn energy(&self) -> f32 {
        self.data.energy
    }
}

impl<'a> Optimizer<Obs<'a>, LMState, Vec6, Vec6, PreEval, LMPartialState, f32> for LMOptimizer {
    fn initial_energy() -> f32 {
        f32::INFINITY
    }

    fn compute_step(state: &LMState) -> Vec6 {
        let mut hessian = state.data.hessian.clone();
        hessian.m11 = (1.0 + state.lm_coef) * hessian.m11;
        hessian.m22 = (1.0 + state.lm_coef) * hessian.m22;
        hessian.m33 = (1.0 + state.lm_coef) * hessian.m33;
        hessian.m44 = (1.0 + state.lm_coef) * hessian.m44;
        hessian.m55 = (1.0 + state.lm_coef) * hessian.m55;
        hessian.m66 = (1.0 + state.lm_coef) * hessian.m66;
        hessian.cholesky().unwrap().solve(&state.data.gradient)
    }

    fn apply_step(delta: Vec6, model: &Vec6) -> Vec6 {
        let delta_warp = Affine2::from_matrix_unchecked(warp_mat(delta));
        let old_warp = Affine2::from_matrix_unchecked(warp_mat(model.clone()));
        warp_params((old_warp * delta_warp.inverse()).unwrap())
    }

    fn pre_eval(obs: &Obs, model: &Vec6) -> PreEval {
        let (nb_rows, _) = obs.template.shape();
        let mut x = 0;
        let mut y = 0;
        let mut inside_indices = Vec::new();
        let mut residuals = Vec::new();
        let mut energy = 0.0;
        for (idx, tmp) in obs.template.iter().enumerate() {
            // check if warp(x,y) is inside the image
            let (u, v) = warp(model, x as f32, y as f32);
            if let Some(im) = interpolate(u, v, &obs.image) {
                // precompute residuals and energy
                let r = im - *tmp as f32;
                energy = energy + r * r;
                residuals.push(r);
                inside_indices.push(idx); // keep only inside points
            }
            // update x and y positions
            y = y + 1;
            if y >= nb_rows {
                x = x + 1;
                y = 0;
            }
        }
        energy = energy / residuals.len() as f32;
        (inside_indices, residuals, energy)
    }

    fn eval(obs: &Obs, energy: f32, pre_eval: PreEval, model: Vec6) -> LMPartialState {
        let (inside_indices, residuals, new_energy) = pre_eval;
        if new_energy > energy {
            Err(new_energy)
        } else {
            let mut gradient = Vec6::zeros();
            let mut hessian = Mat6::zeros();
            for (i, idx) in inside_indices.iter().enumerate() {
                let jac = obs.jacobians[*idx];
                let hes = obs.hessians[*idx];
                let r = residuals[i];
                gradient = gradient + jac * r;
                hessian = hessian + hes;
            }
            Ok(LMData {
                hessian,
                gradient,
                energy: new_energy,
                model,
            })
        }
    }

    fn stop_criterion(nb_iter: usize, s0: LMState, s1: LMPartialState) -> (LMState, Continue) {
        let too_many_iterations = nb_iter > 20;
        match (s1, too_many_iterations) {
            // Max number of iterations reached:
            (Err(_), true) => (s0, Continue::Stop),
            (Ok(data), true) => {
                println!("Energy: {}", data.energy);
                println!("Gradient norm: {}", data.gradient.norm());
                let kept_state = LMState {
                    lm_coef: s0.lm_coef, // does not matter actually
                    data: data,
                };
                (kept_state, Continue::Stop)
            }
            // Can continue to iterate:
            (Err(energy), false) => {
                println!("\t back from: {}", energy);
                let mut kept_state = s0;
                kept_state.lm_coef = 10.0 * kept_state.lm_coef;
                (kept_state, Continue::Forward)
            }
            (Ok(data), false) => {
                let d_energy = s0.data.energy - data.energy;
                let gradient_norm = data.gradient.norm();
                // 1.0 is totally empiric here
                let continuation = if d_energy > 1.0 {
                    Continue::Forward
                } else {
                    Continue::Stop
                };
                println!("Energy: {}", data.energy);
                println!("Gradient norm: {}", gradient_norm);
                let kept_state = LMState {
                    lm_coef: 0.1 * s0.lm_coef,
                    data: data,
                };
                (kept_state, continuation)
            }
        }
    }
}

// Helper ######################################################################

fn gradient(im: &DMatrix<u8>) -> (DMatrix<i16>, DMatrix<i16>) {
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

fn affine_jacobians(grad_x: &DMatrix<i16>, grad_y: &DMatrix<i16>) -> Vec<Vec6> {
    let (nb_rows, _) = grad_x.shape();
    let mut x = 0;
    let mut y = 0;
    let mut jacobians = Vec::new();
    for (gx, gy) in grad_x.iter().zip(grad_y.iter()) {
        let gx_f = *gx as f32;
        let gy_f = *gy as f32;
        let x_f = x as f32;
        let y_f = y as f32;
        let jac = Vec6::new(x_f * gx_f, x_f * gy_f, y_f * gx_f, y_f * gy_f, gx_f, gy_f);
        jacobians.push(jac);
        y = y + 1;
        if y >= nb_rows {
            x = x + 1;
            y = 0;
        }
    }
    jacobians
}

fn hessians_vec(jacobians: &Vec<Vec6>) -> Vec<Mat6> {
    jacobians.iter().map(|j| j * j.transpose()).collect()
}

fn warp(model: &Vec6, x: f32, y: f32) -> (f32, f32) {
    (
        (1.0 + model.x) * x + model.z * y + model.a,
        model.y * x + (1.0 + model.w) * y + model.b,
    )
}

fn interpolate(x: f32, y: f32, image: &DMatrix<u8>) -> Option<f32> {
    let (height, width) = image.shape();
    let u = x.floor();
    let v = y.floor();
    if u >= 0.0 && u < (width - 2) as f32 && v >= 0.0 && v < (height - 2) as f32 {
        let u_0 = u as usize;
        let v_0 = v as usize;
        let u_1 = u_0 + 1;
        let v_1 = v_0 + 1;
        let vu_00 = image[(v_0, u_0)] as f32;
        let vu_10 = image[(v_1, u_0)] as f32;
        let vu_01 = image[(v_0, u_1)] as f32;
        let vu_11 = image[(v_1, u_1)] as f32;
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

fn warp_mat(params: Vec6) -> Mat3 {
    // Affine warp parameterization:
    // [ 1+p1  p3  p5 ]
    // [  p2  1+p4 p6 ]
    #[rustfmt::skip]
    let mat = Mat3::new(
        1.0 + params.x, params.z,       params.a,
        params.y,       1.0 + params.w, params.b,
        0.0,            0.0,            1.0,
    );
    mat
}

fn warp_params(mat: Mat3) -> Vec6 {
    Vec6::new(
        mat.m11 - 1.0,
        mat.m21,
        mat.m12,
        mat.m22 - 1.0,
        mat.m13,
        mat.m23,
    )
}
