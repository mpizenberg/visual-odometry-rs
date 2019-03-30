// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

extern crate image;
extern crate nalgebra as na;
extern crate visual_odometry_rs as vors;

use na::DMatrix;
use std::{env, error::Error, fs, io::BufReader, io::Read, path::Path, path::PathBuf};

use vors::core::camera::Intrinsics;
use vors::core::track::inverse_compositional_robust as track;
use vors::dataset::tum_rgbd;
use vors::misc::{helper, interop};

fn main() {
    let args: Vec<String> = env::args().collect();
    if let Err(error) = my_run(&args) {
        eprintln!("{:?}", error);
    }
}

const USAGE: &str = "Usage: ./vors_track [fr1|fr2|fr3|icl] associations_file";

fn my_run(args: &[String]) -> Result<(), Box<Error>> {
    // Check that the arguments are correct.
    let valid_args = check_args(args)?;

    // Build a vector containing timestamps and full paths of images.
    let associations = parse_associations(valid_args.associations_file_path)?;

    // Setup tracking configuration.
    let config = track::Config {
        nb_levels: 6,
        candidates_diff_threshold: 7,
        depth_scale: tum_rgbd::DEPTH_SCALE,
        intrinsics: valid_args.intrinsics,
        idepth_variance: 0.0001,
    };

    // Initialize tracker with first depth and color image.
    let (depth_map, img) = read_images(&associations[0])?;
    let depth_time = associations[0].depth_timestamp;
    let img_time = associations[0].color_timestamp;
    let mut tracker = config.init(depth_time, &depth_map, img_time, img);

    // Track every frame in the associations file.
    for assoc in associations.iter().skip(1) {
        // Load depth and color images.
        let (depth_map, img) = read_images(assoc)?;

        // Track the rgb-d image.
        tracker.track(
            assoc.depth_timestamp,
            &depth_map,
            assoc.color_timestamp,
            img,
        );

        // Print to stdout the frame pose.
        let (timestamp, pose) = tracker.current_frame();
        println!("{}", (tum_rgbd::Frame { timestamp, pose }).to_string());
    }

    Ok(())
}

struct Args {
    associations_file_path: PathBuf,
    intrinsics: Intrinsics,
}

/// Verify that command line arguments are correct.
fn check_args(args: &[String]) -> Result<Args, String> {
    // eprintln!("{:?}", args);
    if let [_, camera_id, associations_file_path_str] = args {
        let intrinsics = create_camera(camera_id)?;
        let associations_file_path = PathBuf::from(associations_file_path_str);
        if associations_file_path.is_file() {
            Ok(Args {
                intrinsics,
                associations_file_path,
            })
        } else {
            eprintln!("{}", USAGE);
            Err(format!(
                "The association file does not exist or is not reachable: {}",
                associations_file_path_str
            ))
        }
    } else {
        eprintln!("{}", USAGE);
        Err("Wrong number of arguments".to_string())
    }
}

/// Create camera depending on `camera_id` command line argument.
fn create_camera(camera_id: &str) -> Result<Intrinsics, String> {
    match camera_id {
        "fr1" => Ok(tum_rgbd::INTRINSICS_FR1),
        "fr2" => Ok(tum_rgbd::INTRINSICS_FR2),
        "fr3" => Ok(tum_rgbd::INTRINSICS_FR3),
        "icl" => Ok(tum_rgbd::INTRINSICS_ICL_NUIM),
        _ => {
            eprintln!("{}", USAGE);
            Err(format!("Unknown camera id: {}", camera_id))
        }
    }
}

/// Open an association file and parse it into a vector of Association.
fn parse_associations<P: AsRef<Path>>(
    file_path: P,
) -> Result<Vec<tum_rgbd::Association>, Box<Error>> {
    let file = fs::File::open(&file_path)?;
    let mut file_reader = BufReader::new(file);
    let mut content = String::new();
    file_reader.read_to_string(&mut content)?;
    tum_rgbd::parse::associations(&content)
        .map(|v| v.iter().map(|a| abs_path(&file_path, a)).collect())
        .map_err(|s| s.into())
}

/// Transform relative images file paths into absolute ones.
fn abs_path<P: AsRef<Path>>(file_path: P, assoc: &tum_rgbd::Association) -> tum_rgbd::Association {
    let parent = file_path
        .as_ref()
        .parent()
        .expect("How can this have no parent");
    tum_rgbd::Association {
        depth_timestamp: assoc.depth_timestamp,
        depth_file_path: parent.join(&assoc.depth_file_path),
        color_timestamp: assoc.color_timestamp,
        color_file_path: parent.join(&assoc.color_file_path),
    }
}

/// Read a depth and color image given by an association.
fn read_images(assoc: &tum_rgbd::Association) -> Result<(DMatrix<u16>, DMatrix<u8>), Box<Error>> {
    let (w, h, depth_map_vec_u16) = helper::read_png_16bits(&assoc.depth_file_path)?;
    let depth_map = DMatrix::from_row_slice(h, w, depth_map_vec_u16.as_slice());
    let img = interop::matrix_from_image(image::open(&assoc.color_file_path)?.to_luma());
    Ok((depth_map, img))
}
