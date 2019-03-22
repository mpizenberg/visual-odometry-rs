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
    "Usage: cargo run --release --example dataset_tum-read-trajectory trajectory_file";

fn run(args: Vec<String>) -> Result<(), Box<Error>> {
    // Check that the arguments are correct.
    let trajectory_file_path = check_args(args)?;

    // Build a vector of frames, containing timestamps and camera poses.
    let frames = parse_trajectory(trajectory_file_path)?;

    // Print to stdout first few frames.
    println!("First few frames:");
    for frame in frames.iter().take(5) {
        println!("{}", frame.to_string());
    }

    Ok(())
}

/// Verify that command line arguments are correct.
fn check_args(args: Vec<String>) -> Result<PathBuf, String> {
    match args.as_slice() {
        [_, trajectory_file_path_str] => {
            let trajectory_file_path = PathBuf::from(trajectory_file_path_str);
            if trajectory_file_path.is_file() {
                Ok(trajectory_file_path)
            } else {
                eprintln!("{}", USAGE);
                Err(format!(
                    "The trajectory file does not exist or is not reachable: {}",
                    trajectory_file_path_str
                ))
            }
        }
        _ => {
            eprintln!("{}", USAGE);
            Err("Wrong number of arguments".to_string())
        }
    }
}

/// Open a trajectory file and parse it into a vector of Frames.
fn parse_trajectory(file_path: PathBuf) -> Result<Vec<tum_rgbd::Frame>, Box<Error>> {
    let file = fs::File::open(&file_path)?;
    let mut file_reader = BufReader::new(file);
    let mut content = String::new();
    file_reader.read_to_string(&mut content)?;
    tum_rgbd::parse::trajectory(content).map_err(|s| s.into())
}
