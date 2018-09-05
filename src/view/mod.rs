extern crate image;
extern crate nalgebra;

use image::RgbImage;
use nalgebra::DMatrix;

use colormap;
use interop;
use inverse_depth::{self, InverseDepth};

pub type Float = f32;

// Create an RGB image of an inverse depth map.
pub fn color_idepth(idepth_map: &DMatrix<InverseDepth>) -> RgbImage {
    let viridis = &colormap::viridis_u8()[0..256];
    let (d_min, d_max) = min_max(idepth_map).unwrap();
    interop::rgb_from_matrix(&idepth_map.map(|idepth| idepth_color(viridis, d_min, d_max, &idepth)))
}

fn min_max(idepth_map: &DMatrix<InverseDepth>) -> Option<(Float, Float)> {
    let mut min_temp: Float = 10000.0;
    let mut max_temp: Float = 0.0;
    idepth_map.iter().for_each(|idepth| {
        if let Some((d, _)) = inverse_depth::with_variance(idepth) {
            min_temp = min_temp.min(d);
            max_temp = max_temp.max(d);
        }
    });
    Some((min_temp, max_temp))
}

// INVERSE DEPTH ###########################################

// Visualize the enum as an 8-bits intensity:
// Unknown:      black
// Discarded:    gray
// WithVariance: white
pub fn idepth_enum(idepth: &InverseDepth) -> u8 {
    match idepth {
        InverseDepth::Unknown => 0u8,
        InverseDepth::Discarded => 50u8,
        InverseDepth::WithVariance(_, _) => 255u8,
    }
}

// Use viridis colormap + red for Discarded
pub fn idepth_color(
    colormap: &[(u8, u8, u8)],
    d_min: Float,
    d_max: Float,
    idepth: &InverseDepth,
) -> (u8, u8, u8) {
    match idepth {
        InverseDepth::Unknown => (0, 0, 0),
        InverseDepth::Discarded => (255, 0, 0),
        InverseDepth::WithVariance(d, _) => {
            let idx = (255.0 * (d - d_min) / (d_max - d_min)).round() as usize;
            colormap[idx]
        }
    }
}
