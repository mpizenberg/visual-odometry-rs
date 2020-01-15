// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

extern crate image;
extern crate nalgebra as na;
extern crate visual_odometry_rs as vors;

use na::DMatrix;
use std::{env, error::Error, fs, io::BufReader, io::Read, path::Path, path::PathBuf};

use vors::core::camera::Intrinsics;
use vors::core::track::inverse_compositional_norm as track;
use vors::dataset::tum_rgbd;
use vors::misc::{helper, interop};

fn main() {
    let args: Vec<String> = env::args().collect();
    if let Err(error) = my_run(&args) {
        eprintln!("{:?}", error);
    }
}

const USAGE: &str =
    "Usage: cargo run --release --example optim_vo [icl|fr1|fr2|fr3] associations_file n1 n2";

fn my_run(args: &[String]) -> Result<(), Box<dyn Error>> {
    // Check that the arguments are correct.
    let valid_args = check_args(args)?;

    // Build a vector containing timestamps and full paths of images.
    let associations = parse_associations(&valid_args.associations_file_path)?;

    // Setup tracking configuration.
    let config = track::Config {
        nb_levels: 6,
        candidates_diff_threshold: 7,
        depth_scale: tum_rgbd::DEPTH_SCALE,
        intrinsics: valid_args.intrinsics,
        idepth_variance: 0.0001,
    };

    // Initialize tracker with first depth and color image.
    let (depth_map, img) = read_images(&associations[valid_args.img1])?;
    let depth_time = associations[valid_args.img1].depth_timestamp;
    let img_time = associations[valid_args.img1].color_timestamp;
    let mut tracker = config.init(depth_time, &depth_map, img_time, img);

    // Track second frame.
    let assoc_img2 = &associations[valid_args.img2];
    let (depth_map, img) = read_images(assoc_img2)?;

    // Track the rgb-d image.
    tracker.track(
        false,
        assoc_img2.depth_timestamp,
        &depth_map,
        assoc_img2.color_timestamp,
        img,
    );

    // Print to stdout the frame pose.
    let (timestamp, pose) = tracker.current_frame();
    println!("{}", (tum_rgbd::Frame { timestamp, pose }).to_string());

    // Retrieve to ground truth.
    let parent_dir = valid_args.associations_file_path.parent().ok_or("parent")?;
    let trajectory = parse_trajectory(parent_dir.join("trajectory-gt.txt"))?;
    let pose1 = &trajectory[valid_args.img1 - 1].pose;
    let pose2 = &trajectory[valid_args.img2 - 1].pose;
    let motion = pose1.inverse() * pose2;
    println!("{}", motion);

    // Compare to ground truth.
    let motion_error = pose.inverse() * motion;
    println!(
        "Translation error: {} cm",
        100.0 * motion_error.translation.vector.norm()
    );
    println!(
        "Rotation error: {} degrees",
        motion_error.rotation.angle().to_degrees()
    );

    Ok(())
}

struct Args {
    associations_file_path: PathBuf,
    intrinsics: Intrinsics,
    img1: usize,
    img2: usize,
}

/// Verify that command line arguments are correct.
fn check_args(args: &[String]) -> Result<Args, Box<dyn Error>> {
    // eprintln!("{:?}", args);
    if let [_, camera_id, associations_file_path_str, n1, n2] = args {
        let intrinsics = create_camera(camera_id)?;
        let associations_file_path = PathBuf::from(associations_file_path_str);
        if associations_file_path.is_file() {
            let img1 = n1.parse()?;
            let img2 = n2.parse()?;
            Ok(Args {
                intrinsics,
                associations_file_path,
                img1,
                img2,
            })
        } else {
            eprintln!("{}", USAGE);
            Err(format!(
                "The association file does not exist or is not reachable: {}",
                associations_file_path_str
            )
            .into())
        }
    } else {
        eprintln!("{}", USAGE);
        Err("Wrong number of arguments".into())
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
) -> Result<Vec<tum_rgbd::Association>, Box<dyn Error>> {
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
fn read_images(
    assoc: &tum_rgbd::Association,
) -> Result<(DMatrix<u16>, DMatrix<u8>), Box<dyn Error>> {
    let (w, h, depth_map_vec_u16) = helper::read_png_16bits(&assoc.depth_file_path)?;
    let depth_map = DMatrix::from_row_slice(h, w, depth_map_vec_u16.as_slice());
    let img = interop::matrix_from_image(image::open(&assoc.color_file_path)?.to_luma());
    Ok((depth_map, img))
}

/// Open a trajectory file and parse it into a vector of Frames.
fn parse_trajectory(file_path: PathBuf) -> Result<Vec<tum_rgbd::Frame>, Box<dyn Error>> {
    let file = fs::File::open(&file_path)?;
    let mut file_reader = BufReader::new(file);
    let mut content = String::new();
    file_reader.read_to_string(&mut content)?;
    tum_rgbd::parse::trajectory(&content).map_err(|s| s.into())
}
