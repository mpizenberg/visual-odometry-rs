// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

extern crate nalgebra as na;
extern crate visual_odometry_rs as vors;

use std::{env, error::Error, path::Path, path::PathBuf, process::exit};
use vors::core::{candidates::coarse_to_fine as candidates, gradient, multires};
use vors::misc::{interop, view};

type Img = na::DMatrix<u8>;
type Mask = na::DMatrix<bool>;

fn main() {
    let args: Vec<String> = env::args().collect();
    if let Err(error) = run(args) {
        eprintln!("{:?}", error);
        exit(1);
    }
}

const USAGE: &str = "Usage: cargo run --release --example candidates_coarse-to-fine image_file";

fn run(args: Vec<String>) -> Result<(), Box<Error>> {
    // Check that the arguments are correct.
    let image_file_path = check_args(args)?;
    let img = read_image(&image_file_path)?;

    // Compute coarse to fine candidates.
    let nb_levels = 6;
    let diff_threshold = 7;
    let (candidates_coarse_to_fine, multires_img) =
        generate_candidates(img, nb_levels, diff_threshold);

    // Save each level in an image on the disk.
    let parent_dir = image_file_path.parent().unwrap();
    for level in 0..nb_levels {
        let candidates_level = &candidates_coarse_to_fine[nb_levels - level - 1];
        let img_level = &multires_img[level];
        save_candidates(candidates_level, img_level, parent_dir, level)?;
    }

    // Display some stats.
    let nb_candidates = |mask: &Mask| mask.fold(0, |sum, x| if x { sum + 1 } else { sum });
    let nb_candidates_levels: Vec<_> = candidates_coarse_to_fine
        .iter()
        .map(nb_candidates)
        .collect();
    eprintln!("Candidate points per level: {:?}", nb_candidates_levels);

    Ok(())
}

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

// HELPERS #####################################################################

/// Verify that command line arguments are correct.
fn check_args(args: Vec<String>) -> Result<PathBuf, String> {
    match args.as_slice() {
        [_, image_file_path_str] => {
            let image_file_path = PathBuf::from(image_file_path_str);
            if image_file_path.is_file() {
                Ok(image_file_path)
            } else {
                eprintln!("{}", USAGE);
                Err(format!("This file does not exist: {}", image_file_path_str))
            }
        }
        _ => {
            eprintln!("{}", USAGE);
            Err("Wrong number of arguments".to_string())
        }
    }
}

fn read_image<P: AsRef<Path>>(image_path: P) -> Result<Img, Box<Error>> {
    Ok(interop::matrix_from_image(
        image::open(image_path)?.to_luma(),
    ))
}

fn save_candidates<P: AsRef<Path>>(
    candidates_map: &Mask,
    img: &Img,
    dir: P,
    level: usize,
) -> Result<(), std::io::Error> {
    let color_candidates = view::candidates_on_image(img, candidates_map);
    let file_name = format!("candidates_{}.png", level);
    color_candidates.save(dir.as_ref().join(file_name))
}
