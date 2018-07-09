extern crate image;
extern crate nalgebra;

use nalgebra::DMatrix;
use image::RgbImage;

use interop;
use colormap;
use inverse_depth::{self, InverseDepth};


// Create an RGB image of an inverse depth map.
pub fn color_idepth(idepth_map: &DMatrix<InverseDepth>) -> RgbImage {
    let viridis = &colormap::viridis()[0..256];
    let (d_min, d_max) = min_max(idepth_map).unwrap();
    interop::rgb_from_matrix(
        &idepth_map.map(
            |idepth| inverse_depth::to_color(viridis, d_min, d_max, &idepth)
        )
    )
}

fn min_max(idepth_map: &DMatrix<InverseDepth>) -> Option<(f32, f32)> {
    let mut min_temp = 10000.0_f32;
    let mut max_temp = 0.0_f32;
    idepth_map.iter().for_each(|idepth| {
        if let Some((d,_)) = inverse_depth::with_variance(idepth) {
            min_temp = min_temp.min(d);
            max_temp = max_temp.max(d);
        }
    });
    Some((min_temp, max_temp))
}
