// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use visual_odometry_rs as vors;

use image;
use nalgebra as na;
use std::{env, error::Error, fs, io::BufReader, io::Read, path::Path, path::PathBuf};
use vors::core::{candidates::coarse_to_fine as candidates, gradient, multires};
use vors::dataset::tum_rgbd;
use vors::misc::{interop, view};

type Img = na::DMatrix<u8>;
type Mask = na::DMatrix<bool>;

fn main() {
    let args: Vec<String> = env::args().collect();
    if let Err(error) = my_run(&args) {
        eprintln!("{:?}", error);
    }
}

const USAGE: &str = "Usage: ./vors_candidates associations_file out_dir";

fn my_run(args: &[String]) -> Result<(), Box<Error>> {
    // Check that the arguments are correct.
    let (associations_file, out_dir) = check_args(args)?;

    // Build a vector containing timestamps and full paths of images.
    let associations = parse_associations(associations_file)?;

    // Candidates configuration.
    let nb_levels = 6;
    let diff_threshold = 7;

    // Create output directory.
    fs::create_dir_all(&out_dir)?;

    // Compute candidates of every frame in the associations file.
    for assoc in associations.iter() {
        // Load color image.
        let img = read_image(&assoc.color_file_path)?;
        let (candidates, imgs) = generate_candidates(img.clone(), nb_levels, diff_threshold);
        // save_all_candidates(candidates, imgs, &out_dir, assoc.color_timestamp)?;
        save_candidates(
            candidates.last().unwrap(),
            &imgs[0],
            &out_dir,
            assoc.color_timestamp,
            0,
        )?;
    }

    Ok(())
}

// HELPERS #####################################################################

fn generate_candidates(img: Img, nb_levels: usize, diff_threshold: u16) -> (Vec<Mask>, Vec<Img>) {
    // Compute the multi-resolution image pyramid.
    let multires_img = multires::mean_pyramid(nb_levels, img);

    // Compute the multi-resolution gradients pyramid (without first level).
    let mut multires_grad = multires::gradients_squared_norm(&multires_img);

    // Insert the gradient of the original resolution at the first position.
    // This one cannot be computed by the bloc gradients used in multires::gradients_squared_norm.
    multires_grad.insert(0, gradient::squared_norm_direct(&multires_img[0]));

    // Call the coarse to fine candidates selection algorithm.
    let candidates_coarse_to_fine = candidates::select(diff_threshold, &multires_grad);
    (candidates_coarse_to_fine, multires_img)
}

/// Verify that command line arguments are correct.
fn check_args(args: &[String]) -> Result<(PathBuf, PathBuf), String> {
    // eprintln!("{:?}", args);
    if let [_, associations_file_str, out_dir_str] = args {
        let associations_file = PathBuf::from(associations_file_str);
        let out_dir = PathBuf::from(out_dir_str);
        if associations_file.is_file() {
            Ok((associations_file, out_dir))
        } else {
            eprintln!("{}", USAGE);
            Err(format!(
                "The association file does not exist or is not reachable: {}",
                associations_file_str
            ))
        }
    } else {
        eprintln!("{}", USAGE);
        Err("Wrong number of arguments".to_string())
    }
}

fn read_image<P: AsRef<Path>>(image_path: P) -> Result<Img, Box<Error>> {
    Ok(interop::matrix_from_image(
        image::open(image_path)?.to_luma(),
    ))
}

fn save_all_candidates(
    candidates: Vec<Mask>,
    imgs: Vec<Img>,
    out_dir: &PathBuf,
    timestamp: f64,
) -> Result<(), std::io::Error> {
    for (lvl, img) in imgs.iter().enumerate() {
        let mask = &candidates[imgs.len() - lvl - 1];
        save_candidates(mask, img, &out_dir, timestamp, lvl)?;
        save_image(img, &out_dir, timestamp, lvl)?;
    }
    Ok(())
}

fn save_image<P: AsRef<Path>>(
    img: &Img,
    dir: P,
    timestamp: f64,
    level: usize,
) -> Result<(), std::io::Error> {
    let file_name = format!("img_{}_lvl_{}.png", timestamp, level);
    interop::image_from_matrix(img).save(dir.as_ref().join(file_name))
}

fn save_candidates<P: AsRef<Path>>(
    candidates_map: &Mask,
    img: &Img,
    dir: P,
    timestamp: f64,
    level: usize,
) -> Result<(), std::io::Error> {
    let color_candidates = view::candidates_on_image(img, candidates_map);
    let file_name = format!("candidates_{}_lvl_{}.png", timestamp, level);
    color_candidates.save(dir.as_ref().join(file_name))
}

/// Open an association file and parse it into a vector of Association.
fn parse_associations(file_path: PathBuf) -> Result<Vec<tum_rgbd::Association>, Box<Error>> {
    let file = fs::File::open(&file_path)?;
    let mut file_reader = BufReader::new(file);
    let mut content = String::new();
    file_reader.read_to_string(&mut content)?;
    tum_rgbd::parse::associations(&content)
        .map(|v| v.iter().map(|a| abs_path(&file_path, a)).collect())
        .map_err(|s| s.into())
}

/// Transform relative images file paths into absolute ones.
fn abs_path(file_path: &PathBuf, assoc: &tum_rgbd::Association) -> tum_rgbd::Association {
    let parent = file_path.parent().expect("How can this have no parent?");
    tum_rgbd::Association {
        depth_timestamp: assoc.depth_timestamp,
        depth_file_path: parent.join(&assoc.depth_file_path),
        color_timestamp: assoc.color_timestamp,
        color_file_path: parent.join(&assoc.color_file_path),
    }
}
