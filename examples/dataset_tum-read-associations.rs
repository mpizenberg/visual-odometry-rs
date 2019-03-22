// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

extern crate visual_odometry_rs as vors;

use std::{env, error::Error, fs, io::BufReader, io::Read, path::PathBuf, process::exit};
use vors::dataset::tum_rgbd;

fn main() {
    let args: Vec<String> = env::args().collect();
    if let Err(error) = run(args) {
        eprintln!("{:?}", error);
        exit(1);
    }
}

const USAGE: &str =
    "Usage: cargo run --release --example dataset_tum-read-associations associations_file";

fn run(args: Vec<String>) -> Result<(), Box<Error>> {
    // Check that the arguments are correct.
    let associations_file_path = check_args(args)?;

    // Build a vector containing timestamps and full paths of images.
    let associations = parse_associations(associations_file_path)?;

    // Print to stdout first few associations.
    println!("First 3 associations:");
    for assoc in associations.iter().take(3) {
        println!("");
        println!("Depth image timestamp: {}", assoc.depth_timestamp);
        println!(
            "Depth image absolute path: {}",
            assoc.depth_file_path.display()
        );
        println!("Color image timestamp: {}", assoc.color_timestamp);
        println!(
            "Color image absolute path: {}",
            assoc.color_file_path.display()
        );
    }

    Ok(())
}

/// Verify that command line arguments are correct.
fn check_args(args: Vec<String>) -> Result<PathBuf, String> {
    match args.as_slice() {
        [_, associations_file_path_str] => {
            let associations_file_path = PathBuf::from(associations_file_path_str);
            if associations_file_path.is_file() {
                Ok(associations_file_path)
            } else {
                eprintln!("{}", USAGE);
                Err(format!(
                    "The association file does not exist or is not reachable: {}",
                    associations_file_path_str
                ))
            }
        }
        _ => {
            eprintln!("{}", USAGE);
            Err("Wrong number of arguments".to_string())
        }
    }
}

/// Open an association file and parse it into a vector of Association.
fn parse_associations(file_path: PathBuf) -> Result<Vec<tum_rgbd::Association>, Box<Error>> {
    let file = fs::File::open(&file_path)?;
    let mut file_reader = BufReader::new(file);
    let mut content = String::new();
    file_reader.read_to_string(&mut content)?;
    tum_rgbd::parse::associations(content)
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
