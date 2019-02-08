extern crate computer_vision_rs as cv;
extern crate csv;
extern crate image;
extern crate nalgebra as na;

use std::fs;

use cv::camera::Camera;
use cv::candidates;
use cv::eval;
use cv::helper;
use cv::icl_nuim;
use cv::inverse_depth::{self, InverseDepth};
use cv::multires;

use na::DMatrix;

const OUT_DIR: &str = "out/example/multires_depth_strategy/";
type Float = f32;

// #[allow(dead_code)]
fn main() {
    // Config of images to test.
    let start_img = 40;
    let nb_img = 60;
    let nb_reprojections = 10;
    let end_img_reproj = start_img + nb_img + nb_reprojections + 1;

    // Put all necessary images and depth maps in memory.
    println!("Loading images");
    let (multires_imgs, depth_maps): (Vec<_>, Vec<_>) = (start_img..end_img_reproj)
        .map(|id| icl_nuim::open_imgs(id).unwrap())
        .map(|(img, depth_map)| (multires::mean_pyramid(6, img), depth_map))
        .unzip();
    println!("Images loaded");

    // Read trajectory file with TUM RGBD syntax (same as icl nuim syntax)
    let all_extrinsics = icl_nuim::read_extrinsics("data/trajectory-gt.txt").unwrap();
    let multires_cameras = &all_extrinsics[(start_img - 1)..(end_img_reproj - 1)]
        .into_iter()
        .map(|ext| Camera::new(icl_nuim::INTRINSICS, *ext))
        .map(|cam| cam.multi_res(6))
        .collect::<Vec<_>>();

    // Evaluate all strategies.
    let mut evaluations = Vec::new();
    for id_img in 0..nb_img {
        let multires_img = &multires_imgs[id_img];
        let depth_map = &depth_maps[id_img];
        let multires_cam = &multires_cameras[id_img];
        let mut sub_eval = Vec::new();
        for reproj in 1..=nb_reprojections {
            let multires_img_reproj = &multires_imgs[id_img + reproj];
            let multires_cam_reproj = &multires_cameras[id_img + reproj];
            sub_eval.push(eval_all_strats(
                depth_map,
                multires_img,
                multires_img_reproj,
                multires_cam,
                multires_cam_reproj,
            ));
        }
        evaluations.push(sub_eval);
    }

    // Put evaluation results into a CSV file.
    println!("Exporting to CSV");
    fs::create_dir_all(OUT_DIR).unwrap();
    let csv_file = &[OUT_DIR, "eval.csv"].concat();
    let csv_header = &[
        "reprojection_error",
        "id_image",
        "id_reprojected",
        "strat",
        "level",
    ];
    let mut writer = csv::Writer::from_path(csv_file).unwrap();
    writer.write_record(csv_header).unwrap();
    for id_img in 0..evaluations.len() {
        for id_reproj in 0..evaluations[id_img].len() {
            let reproj_errors = &evaluations[id_img][id_reproj];
            let errors_stat = &reproj_errors.0;
            let errors_dso = &reproj_errors.1;
            for id_level in 0..errors_stat.len() {
                let fields_stat = &[
                    errors_stat[id_level].to_string(),
                    id_img.to_string(),
                    id_reproj.to_string(),
                    "stat".to_string(),
                    id_level.to_string(),
                ];
                let fields_dso = &[
                    errors_dso[id_level].to_string(),
                    id_img.to_string(),
                    id_reproj.to_string(),
                    "dso".to_string(),
                    id_level.to_string(),
                ];
                writer.write_record(fields_stat).unwrap();
                writer.write_record(fields_dso).unwrap();
            }
        }
    }
}

fn eval_all_strats(
    depth_map: &DMatrix<u16>,
    multires_img: &Vec<DMatrix<u8>>,
    multires_img_reproj: &Vec<DMatrix<u8>>,
    multires_cam: &Vec<Camera>,
    multires_cam_reproj: &Vec<Camera>,
) -> (Vec<f32>, Vec<f32>) {
    // Compute candidates masks.
    let multires_gradients_squared_norm = multires::gradients_squared_norm(multires_img);
    let diff_threshold = 7;
    let multires_candidates = candidates::select(diff_threshold, &multires_gradients_squared_norm);
    let candidates = multires_candidates.last().unwrap();

    // Reproject candidates at each resolution for each strategy.
    (
        compute_reprojection_errors(
            depth_map,
            &candidates,
            multires_img,
            multires_img_reproj,
            multires_cam,
            multires_cam_reproj,
            inverse_depth::strategy_statistically_similar,
        ),
        compute_reprojection_errors(
            depth_map,
            &candidates,
            multires_img,
            multires_img_reproj,
            multires_cam,
            multires_cam_reproj,
            inverse_depth::strategy_dso_mean,
        ),
    )
}

fn compute_reprojection_errors<F>(
    depth_map: &DMatrix<u16>,
    candidates: &DMatrix<bool>,
    multires_img: &Vec<DMatrix<u8>>,
    multires_img_reproj: &Vec<DMatrix<u8>>,
    multires_cam: &Vec<Camera>,
    multires_cam_reproj: &Vec<Camera>,
    strategy: F,
) -> Vec<f32>
where
    F: Fn(Vec<(Float, Float)>) -> InverseDepth,
{
    // Compute half resolution inverse depth map.
    let fuse = |a, b, c, d| inverse_depth::fuse(a, b, c, d, &strategy);
    let from_depth = |depth| inverse_depth::from_depth(icl_nuim::DEPTH_SCALE, depth);
    let half_res_idepth = multires::halve(depth_map, |a, b, c, d| {
        let a = from_depth(a);
        let b = from_depth(b);
        let c = from_depth(c);
        let d = from_depth(d);
        fuse(a, b, c, d)
    })
    .unwrap();

    // Use the candidates mask to keep only inverse depth info on candidates points.
    let idepth_candidates = helper::zip_mask_map(
        &half_res_idepth,
        candidates,
        InverseDepth::Unknown,
        |idepth| idepth,
    );

    // Use the "fusing" strategy to generate a multi resolution inverse depth sparse map.
    let multires_idepth = multires::limited_sequence(
        5,
        idepth_candidates,
        |mat| mat,
        |mat| multires::halve(&mat, fuse),
    );

    // Reproject candidates at each resolution.
    let mut errors = Vec::new();
    multires_idepth
        .iter()
        .enumerate()
        .for_each(|(level, idepth)| {
            let cam = &multires_cam[level + 1];
            let cam_reproj = &multires_cam_reproj[level + 1];
            let img = &multires_img[level + 1];
            let img_reproj = &multires_img_reproj[level + 1];
            errors.push(eval::reprojection_error(
                idepth, cam, cam_reproj, img, img_reproj,
            ));
        });
    errors
}
