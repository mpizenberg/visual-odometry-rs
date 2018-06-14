extern crate image;

// #[allow(dead_code)]
fn main() {
    let img_path = "data/images/0001.png";
    let out_path_str = "out/resize_gray.png";
    let img = image::open(img_path).expect("Cannot open image");

    // flip image along horizontal axis
    // let _ = img.fliph()
    //     .save(out_path_str)
    //     .expect("Saving image failed");

    // resize image to grayscale lower resolution
    img.resize(320, 240, image::FilterType::Triangle)
        .grayscale()
        .save(out_path_str)
        // .write_to(&mut out_file, image::PNG)
        .expect("Saving image failed");
}
