// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

extern crate nalgebra as na;
extern crate visual_odometry_rs as vors;

use std::{env, error::Error, path::Path, path::PathBuf, process::exit};
use vors::core::{candidates::dso as candidates, gradient};
use vors::misc::{interop, view};

type Img = na::DMatrix<u8>;

fn main() {
    let args: Vec<String> = env::args().collect();
    if let Err(error) = run(args) {
        eprintln!("{:?}", error);
        exit(1);
    }
}

const USAGE: &str = "Usage: cargo run --release --example candidates_dso image_file";

fn run(args: Vec<String>) -> Result<(), Box<Error>> {
    // Check that the arguments are correct.
    let image_file_path = check_args(args)?;
    let img = read_image(&image_file_path)?;

    // Compute candidates points.
    let candidate_points = generate_candidates(&img);
    save_candidates(&candidate_points, &img, image_file_path.parent().unwrap())?;

    // Display some stats.
    let nb_candidates = candidate_points.fold(0, |sum, x| if x { sum + 1 } else { sum });
    println!("Number of candidate points: {}", nb_candidates);

    Ok(())
}

fn generate_candidates(img: &Img) -> na::DMatrix<bool> {
    // Compute gradients norm of the image.
    let gradients = gradient::squared_norm_direct(img).map(|g2| (g2 as f32).sqrt() as u16);

    // Example of how to adapt default parameters config.
    let mut recursive_config = candidates::DEFAULT_RECURSIVE_CONFIG;
    let mut block_config = candidates::DEFAULT_BLOCK_CONFIG;
    recursive_config.nb_iterations_left = 2;
    block_config.nb_levels = 3;
    block_config.threshold_factor = 0.5;

    // Choose candidates based on gradients norms.
    candidates::select(
        &gradients,
        candidates::DEFAULT_REGION_CONFIG,
        block_config,
        recursive_config,
        2000,
    )
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
    candidates_map: &na::DMatrix<bool>,
    img: &Img,
    dir: P,
) -> Result<(), std::io::Error> {
    let color_candidates = view::candidates_on_image(img, candidates_map);
    color_candidates.save(dir.as_ref().join("candidates.png"))
}
