extern crate image;

use std::fs::File;
use std::path::Path;

// #[allow(dead_code)]
fn main() {
    let img_path = Path::new("data/images/0001.png");
    let img = image::open(&img_path).expect("Cannot open image");
    let out_path = Path::new("out/resize_gray.png");
    let mut out_file = File::create(&out_path).expect("Cannot create file");

    // flip image along horizontal axis
    // let _ = img.fliph()
    //     .save(&mut out_file, image::PNG)
    //     .expect("Saving image failed");

    // resize image to grayscale lower resolution
    let _ = img.resize(320, 240, image::FilterType::Triangle)
        .grayscale()
        .save(&mut out_file, image::PNG)
        .expect("Saving image failed");
}