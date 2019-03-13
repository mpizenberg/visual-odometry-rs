extern crate nalgebra as na;
extern crate rand;
extern crate visual_odometry_rs as vors;

// use rand::{rngs::StdRng, Rng, SeedableRng};
use rand::Rng;
use std::{env, error::Error, f32::consts, path::Path, path::PathBuf, process::exit};
use vors::misc::type_aliases::{Mat6, Vec6};
use vors::{core::gradient, core::multires, misc::interop};
// use vors::math::optimizer::{Continue, OptimizerState};

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
type Vec3 = na::Vector3<f32>;

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
    let img_multires = multires::mean_pyramid(5, img);
    let template_multires = multires::mean_pyramid(5, template);
    let grad_multires: Vec<_> = template_multires.iter().map(gradient::centered).collect();
    let jacobians_multires: Vec<_> = grad_multires.iter().map(affine_jacobians).collect();
    let hessians_multires: Vec<_> = jacobians_multires.iter().map(hessians_vec).collect();
    Ok(())
}

// OPTIMIZER ###################################################################

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
    let m: Mat2 = Mat2::new(s_r * a.cos(), -s_c * a.sin(), s_r * a.sin(), s_c * a.cos());

    // Project points to find suitable random translation range,
    // such that the template stays fully inside the image.
    #[rustfmt::skip]
    let corners = Mat24::new(
        0.0, 0.0, img_rows-1.0, img_rows-1.0,
        0.0, img_cols-1.0, img_cols-1.0, 0.0,
    );
    let t_corners = m * corners;
    let r_min = t_corners.row(0).min();
    let r_max = t_corners.row(0).max();
    let c_min = t_corners.row(1).min();
    let c_max = t_corners.row(1).max();

    // Generate suitable random translation.
    let t_rows_max = (-r_min + 1e-6).max(img_rows - 1.0 - r_max);
    let t_cols_max = (-c_min + 1e-6).max(img_cols - 1.0 - c_max);
    let t_rows = rng.gen_range(-r_min, t_rows_max);
    let t_cols = rng.gen_range(-c_min, t_cols_max);

    // Build the affine transformation matrix.
    let affine2d = Mat23::new(m.m11, m.m12, t_rows, m.m21, m.m22, t_cols);
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

fn project(img: &Img, shape: (usize, usize), affine2d: &Mat23) -> Img {
    let (rows, cols) = shape;
    let project_ij = |i, j| affine2d * Vec3::new(i as f32, j as f32, 1.0);
    Img::from_fn(rows, cols, |i, j| interpolate(&img, project_ij(i, j)))
}

/// Interpolate a pixel in the image.
fn interpolate(img: &Img, t_ij: Vec2) -> u8 {
    // Bilinear interpolation, points are supposed to be fully inside img.
    let (x, y) = (t_ij.x, t_ij.y);
    let row = x.floor() as usize;
    let col = y.floor() as usize;
    let a = x - x.floor();
    let b = y - y.floor();
    let color = a * b * img[(row + 1, col + 1)] as f32
        + a * (1.0 - b) * img[(row + 1, col)] as f32
        + (1.0 - a) * (1.0 - b) * img[(row, col)] as f32
        + (1.0 - a) * b * img[(row, col + 1)] as f32;
    color as u8
}
