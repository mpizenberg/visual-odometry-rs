// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

extern crate nalgebra as na;
extern crate rand;
extern crate visual_odometry_rs as vors;

// use rand::{rngs::StdRng, Rng, SeedableRng};
use rand::Rng;
use std::{env, error::Error, f32::consts, path::Path, path::PathBuf, process::exit};
use vors::math::optimizer::{Continue, OptimizerState};
use vors::misc::type_aliases::{Mat3, Mat6, Vec3, Vec6};
use vors::{core::gradient, core::multires, misc::interop};

/// In this example, we attempt to find the affine 2D transformation
/// between a template and another image.
///
/// For the purpose of demonstration, we randomly generate ourselve
/// by extraction a square template inside the chosen image.

const USAGE: &str = "Usage: cargo run --release --example optim_affine-2d image_file";

type Mat2 = na::Matrix2<f32>;
type Mat24 = na::Matrix2x4<f32>;
type Mat23 = na::Matrix2x3<f32>;
type Img = na::DMatrix<u8>;
type Vec2 = na::Vector2<f32>;

fn main() {
    let args: Vec<String> = env::args().collect();
    if let Err(err) = run(args) {
        eprintln!("{:?}", err);
        exit(1);
    };
}

fn run(args: Vec<String>) -> Result<(), Box<Error>> {
    // Load image.
    let image_path = check_args(args)?;
    let img = read_image(&image_path)?;

    // Extract template.
    let (template, affine2d) = random_template(&img);
    save_template(&template, image_path.parent().unwrap())?;

    // Precompute multi-resolution observations data.
    // We want roughly 200 points at the lowest resolution.
    let nb_levels = std::cmp::max(
        1,
        (1.0 + (img.len() as f32 / 200.0).log(4.0)).round() as usize,
    );
    let img_multires = multires::mean_pyramid(nb_levels, img);
    let template_multires = multires::mean_pyramid(nb_levels, template);
    let grad_multires: Vec<_> = template_multires.iter().map(gradient::centered).collect();
    let jacobians_multires: Vec<_> = grad_multires.iter().map(affine_jacobians).collect();
    let hessians_multires: Vec<_> = jacobians_multires.iter().map(hessians_vec).collect();

    // Multi-resolution optimization.
    let mut model = Vec6::zeros();
    for level in (0..nb_levels).rev() {
        println!("---------------- Level {}:", level);
        model[4] = 2.0 * model[4];
        model[5] = 2.0 * model[5];
        let obs = Obs {
            template: &template_multires[level],
            image: &img_multires[level],
            jacobians: &jacobians_multires[level],
            hessians: &hessians_multires[level],
        };
        let (final_state, _nb_iter) = LMOptimizerState::iterative_solve(&obs, model)?;
        model = final_state.eval_data.model;
    }

    // Display results.
    println!("Ground truth: {}", affine2d);
    println!("Computed:     {}", warp_mat(model));
    Ok(())
}

// OPTIMIZER ###################################################################

struct Obs<'a> {
    template: &'a Img,
    image: &'a Img,
    jacobians: &'a Vec<Vec6>,
    hessians: &'a Vec<Mat6>,
}

struct LMOptimizerState {
    lm_coef: f32,
    eval_data: EvalData,
}

struct EvalData {
    model: Vec6,
    energy: f32,
    gradient: Vec6,
    hessian: Mat6,
}

type EvalState = Result<EvalData, f32>;

impl LMOptimizerState {
    /// Evaluate energy associated with a model.
    fn eval_energy(obs: &Obs, model: Vec6) -> (f32, Vec<f32>, Vec<usize>) {
        let nb_pixels = obs.template.len();
        let (nb_rows, _) = obs.template.shape();
        let mut x = 0; // column "j"
        let mut y = 0; // row "i"
        let mut inside_indices = Vec::with_capacity(nb_pixels);
        let mut residuals = Vec::with_capacity(nb_pixels);
        let mut energy = 0.0;
        for (idx, tmp) in obs.template.iter().enumerate() {
            let (u, v) = warp(&model, x as f32, y as f32);
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
        energy = energy / inside_indices.len() as f32;
        (energy, residuals, inside_indices)
    }

    /// Compute evaluation data for the next iteration step.
    fn compute_eval_data(obs: &Obs, model: Vec6, pre: (f32, Vec<f32>, Vec<usize>)) -> EvalData {
        let (energy, residuals, inside_indices) = pre;
        let mut gradient = Vec6::zeros();
        let mut hessian = Mat6::zeros();
        for (i, &idx) in inside_indices.iter().enumerate() {
            let jac = obs.jacobians[idx];
            let hes = obs.hessians[idx];
            let res = residuals[i];
            gradient = gradient + jac * res;
            hessian = hessian + hes;
        }
        EvalData {
            model,
            energy,
            gradient,
            hessian,
        }
    }
} // LMOptimizerState

impl<'a> OptimizerState<Obs<'a>, EvalState, Vec6, String> for LMOptimizerState {
    /// Initialize the optimizer state.
    /// Levenberg-Marquardt coefficient start at 0.1.
    fn init(obs: &Obs, model: Vec6) -> Self {
        Self {
            lm_coef: 0.1,
            eval_data: Self::compute_eval_data(obs, model, Self::eval_energy(obs, model)),
        }
    }

    /// Compute the Levenberg-Marquardt step.
    fn step(&self) -> Result<Vec6, String> {
        let mut hessian = self.eval_data.hessian.clone();
        hessian.m11 = (1.0 + self.lm_coef) * hessian.m11;
        hessian.m22 = (1.0 + self.lm_coef) * hessian.m22;
        hessian.m33 = (1.0 + self.lm_coef) * hessian.m33;
        hessian.m44 = (1.0 + self.lm_coef) * hessian.m44;
        hessian.m55 = (1.0 + self.lm_coef) * hessian.m55;
        hessian.m66 = (1.0 + self.lm_coef) * hessian.m66;
        let cholesky = hessian.cholesky().ok_or("Error in cholesky.")?;
        let delta = cholesky.solve(&self.eval_data.gradient);
        let delta_warp = warp_mat(delta);
        let old_warp = warp_mat(self.eval_data.model);
        Ok(warp_params(old_warp * delta_warp.try_inverse().unwrap()))
    }

    /// Evaluate the new model.
    fn eval(&self, obs: &Obs, model: Vec6) -> EvalState {
        let pre = Self::eval_energy(obs, model);
        let energy = pre.0;
        let old_energy = self.eval_data.energy;
        if energy > old_energy {
            Err(energy)
        } else {
            Ok(Self::compute_eval_data(obs, model, pre))
        }
    }

    /// Decide if iterations should continue.
    fn stop_criterion(self, nb_iter: usize, eval_state: EvalState) -> (Self, Continue) {
        let too_many_iterations = nb_iter >= 20;
        match (eval_state, too_many_iterations) {
            // Max number of iterations reached:
            (Err(_), true) => (self, Continue::Stop),
            (Ok(eval_data), true) => {
                println!("energy = {}", eval_data.energy);
                let mut kept_state = self;
                kept_state.eval_data = eval_data;
                (kept_state, Continue::Stop)
            }
            // Max number of iterations not reached yet:
            (Err(energy), false) => {
                let mut kept_state = self;
                kept_state.lm_coef = 10.0 * kept_state.lm_coef;
                println!("\t back from {}, lm_coef = {}", energy, kept_state.lm_coef);
                (kept_state, Continue::Forward)
            }
            (Ok(eval_data), false) => {
                println!("energy = {}", eval_data.energy);
                let delta_energy = self.eval_data.energy - eval_data.energy;
                let mut kept_state = self;
                kept_state.lm_coef = 0.1 * kept_state.lm_coef;
                kept_state.eval_data = eval_data;
                let continuation = if delta_energy > 0.01 {
                    Continue::Forward
                } else {
                    Continue::Stop
                };
                (kept_state, continuation)
            }
        }
    }
}

// HELPERS #####################################################################

fn check_args(args: Vec<String>) -> Result<PathBuf, Box<Error>> {
    if args.len() != 2 {
        eprintln!("{}", USAGE);
        Err("Wrong number of arguments".into())
    } else {
        let image_path = PathBuf::from(&args[1]);
        if image_path.is_file() {
            Ok(image_path)
        } else {
            Err(format!("File does not exist: {}", image_path.display()).into())
        }
    }
}

fn read_image<P: AsRef<Path>>(image_path: P) -> Result<Img, Box<Error>> {
    Ok(interop::matrix_from_image(
        image::open(image_path)?.to_luma(),
    ))
}

fn save_template<P: AsRef<Path>>(template: &Img, dir: P) -> Result<(), std::io::Error> {
    let img = interop::image_from_matrix(template);
    img.save(dir.as_ref().join("template.png"))
}

fn random_template(img: &Img) -> (Img, Mat23) {
    // let seed = [0; 32];
    // let mut rng: StdRng = SeedableRng::from_seed(seed);
    let mut rng = rand::thread_rng();

    // Random scaling.
    let s_r = rng.gen_range(0.7, 0.8);
    let s_c = rng.gen_range(0.7, 0.8);

    // Deduce the size of the template image.
    let (img_rows, img_cols) = img.shape();
    let img_rows = img_rows as f32;
    let img_cols = img_cols as f32;
    let tmp_rows = (s_r * img_rows).floor();
    let tmp_cols = (s_c * img_cols).floor();

    // Random small rotation angle.
    let max_angle = max_template_angle(
        img_rows - 2.0, // add small margin
        img_cols - 2.0,
        tmp_rows,
        tmp_cols,
    );
    let a = rng.gen_range(-max_angle, max_angle);

    // Generate rotation and scaling matrix.
    let m: Mat2 = Mat2::new(s_c * a.cos(), -s_r * a.sin(), s_c * a.sin(), s_r * a.cos());

    // Project points to find suitable random translation range,
    // such that the template stays fully inside the image.
    #[rustfmt::skip]
    let corners = Mat24::new(
        0.0, img_cols-1.0, img_cols-1.0, 0.0,
        0.0, 0.0, img_rows-1.0, img_rows-1.0,
    );
    let t_corners = m * corners;
    let col_min = t_corners.row(0).min();
    let col_max = t_corners.row(0).max();
    let row_min = t_corners.row(1).min();
    let row_max = t_corners.row(1).max();

    // Generate suitable random translation.
    let t_cols_max = (-col_min + 1e-6).max(img_cols - 1.0 - col_max);
    let t_rows_max = (-row_min + 1e-6).max(img_rows - 1.0 - row_max);
    let t_cols = rng.gen_range(-col_min, t_cols_max);
    let t_rows = rng.gen_range(-row_min, t_rows_max);

    // Build the affine transformation matrix.
    let affine2d = Mat23::new(m.m11, m.m12, t_cols, m.m21, m.m22, t_rows);
    println!("affine2d: {}", affine2d);

    // Generate template image with this affine transformation.
    let template = project(&img, (img_rows as usize, img_cols as usize), &affine2d);
    (template, affine2d)
}

/// Find the max rotation of the inner (template) rectangle (rt, ct) such that it stays
/// fully inside the outer (image) rectangle (ri, ci).
/// We suppose rt < ri and ct < ci.
/// If the circumscribed circle of the inner rectangle is fully inside the outer rectangle,
/// all rotation angles are allowed, otherwise, we have to find the limit.
fn max_template_angle(ri: f32, ci: f32, rt: f32, ct: f32) -> f32 {
    // By default, we limit at pi/8.
    let mut threshold = consts::FRAC_PI_8;

    // Inner diagonal.
    let inner_diag = (rt.powi(2) + ct.powi(2)).sqrt();

    // Check rows dimension.
    if inner_diag > ri {
        threshold = threshold.min((ri / inner_diag).asin() - (rt / inner_diag).asin());
    }

    // Check columns dimension.
    if inner_diag > ci {
        threshold = threshold.min((ci / inner_diag).asin() - (ct / inner_diag).asin());
    }

    threshold
}

fn warp(model: &Vec6, x: f32, y: f32) -> (f32, f32) {
    (
        (1.0 + model[0]) * x + model[2] * y + model[4],
        model[1] * x + (1.0 + model[3]) * y + model[5],
    )
}

/// Affine warp parameterization:
/// [ 1+p1  p3  p5 ]
/// [  p2  1+p4 p6 ]
/// [  0    0   1  ]
fn warp_mat(params: Vec6) -> Mat3 {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    Mat3::new(
        params[0] + 1.0, params[2],       params[4],
        params[1],       params[3] + 1.0, params[5],
        0.0,             0.0,             1.0,
    )
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

fn project(img: &Img, shape: (usize, usize), affine2d: &Mat23) -> Img {
    let (rows, cols) = shape;
    let project_ij = |i, j| affine2d * Vec3::new(j as f32, i as f32, 1.0);
    Img::from_fn(rows, cols, |i, j| interpolate_u8(&img, project_ij(i, j)))
}

/// Interpolate a pixel in the image.
/// Bilinear interpolation, points are supposed to be fully inside img.
fn interpolate_u8(img: &Img, pixel: Vec2) -> u8 {
    interpolate(pixel.x, pixel.y, img).unwrap() as u8
}

fn interpolate(x: f32, y: f32, image: &Img) -> Option<f32> {
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

fn affine_jacobians(grad: &(na::DMatrix<i16>, na::DMatrix<i16>)) -> Vec<Vec6> {
    let (grad_x, grad_y) = grad;
    let (nb_rows, _) = grad_x.shape();
    let mut x = 0;
    let mut y = 0;
    let mut jacobians = Vec::with_capacity(grad_x.len());
    for (&gx, &gy) in grad_x.iter().zip(grad_y.iter()) {
        let gx_f = gx as f32;
        let gy_f = gy as f32;
        let x_f = x as f32;
        let y_f = y as f32;
        // CF Baker and Matthews.
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
