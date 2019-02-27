extern crate computer_vision_rs as cv;
extern crate image;
extern crate nalgebra as na;

use na::DMatrix;
use std::{env, error::Error, fs, io::BufReader, io::Read, path::PathBuf};

use cv::camera::Intrinsics;
use cv::helper;
use cv::interop;
use cv::track;
use cv::tum_rgbd;

fn main() {
    let args: Vec<String> = env::args().collect();
    if let Err(error) = my_run(args) {
        eprintln!("{:?}", error);
    }
}

const USAGE: &str = "Usage: ./cvrs_track [fr1|fr2|fr3|icl] associations_file";

fn my_run(args: Vec<String>) -> Result<(), Box<Error>> {
    // Check that the arguments are correct.
    let valid_args = check_args(args)?;

    // Build a vector containing timestamps and full paths of images.
    let associations = parse_associations(valid_args.associations_file_path)?;

    // Setup tracking configuration.
    let config = track::Config {
        nb_levels: 6,
        candidates_diff_threshold: 7,
        depth_scale: 5000.0,
        intrinsics: valid_args.intrinsics,
    };

    // Initialize tracker with first depth and color image.
    let (depth_map, img) = read_images(&associations[0])?;
    let depth_time = associations[0].depth_timestamp;
    let img_time = associations[0].color_timestamp;
    let mut tracker = config.init(depth_time, depth_map, img_time, img);

    // Track every frame in the associations file.
    for assoc in associations.iter().skip(1) {
        // Load depth and color images.
        let (depth_map, img) = read_images(assoc)?;

        // Track the rgb-d image.
        tracker.track(assoc.depth_timestamp, depth_map, assoc.color_timestamp, img);

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

fn check_args(args: Vec<String>) -> Result<Args, String> {
    // eprintln!("{:?}", args);
    match args.as_slice() {
        [_, camera_id, associations_file_path_str] => {
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
        }
        _ => {
            eprintln!("{}", USAGE);
            Err("Wrong number of arguments".to_string())
        }
    }
}

fn create_camera(camera_id: &str) -> Result<Intrinsics, String> {
    match camera_id {
        "fr1" => Ok(Intrinsics {
            principal_point: (318.643040, 255.313989),
            focal_length: 1.0,
            scaling: (517.306408, 516.469215),
            skew: 0.0,
        }),
        "fr2" => Ok(Intrinsics {
            principal_point: (325.141442, 249.701764),
            focal_length: 1.0,
            scaling: (520.908620, 521.007327),
            skew: 0.0,
        }),
        "fr3" => Ok(Intrinsics {
            principal_point: (320.106653, 247.632132),
            focal_length: 1.0,
            scaling: (535.433105, 539.212524),
            skew: 0.0,
        }),
        "icl" => Ok(Intrinsics {
            principal_point: (319.5, 239.5),
            focal_length: 1.0,
            scaling: (481.2, -480.0),
            skew: 0.0,
        }),
        _ => {
            eprintln!("{}", USAGE);
            Err(format!("Unknown camera id: {}", camera_id))
        }
    }
}

fn parse_associations(file_path: PathBuf) -> Result<Vec<tum_rgbd::Association>, Box<Error>> {
    let file = fs::File::open(&file_path)?;
    let mut file_reader = BufReader::new(file);
    let mut content = String::new();
    file_reader.read_to_string(&mut content)?;
    tum_rgbd::parse::associations(content)
        .map(|v| v.iter().map(|a| abs_path(&file_path, a)).collect())
        .map_err(|s| s.into())
}

fn abs_path(file_path: &PathBuf, assoc: &tum_rgbd::Association) -> tum_rgbd::Association {
    let parent = file_path.parent().expect("How can this have no parent");
    tum_rgbd::Association {
        depth_timestamp: assoc.depth_timestamp,
        depth_file_path: parent.join(&assoc.depth_file_path),
        color_timestamp: assoc.color_timestamp,
        color_file_path: parent.join(&assoc.color_file_path),
    }
}

fn read_images(assoc: &tum_rgbd::Association) -> Result<(DMatrix<u16>, DMatrix<u8>), Box<Error>> {
    let (w, h, depth_map_vec_u16) = helper::read_png_16bits(&assoc.depth_file_path)?;
    let depth_map = DMatrix::from_row_slice(h, w, depth_map_vec_u16.as_slice());
    let img = interop::matrix_from_image(image::open(&assoc.color_file_path)?.to_luma());
    Ok((depth_map, img))
}
