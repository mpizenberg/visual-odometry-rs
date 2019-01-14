extern crate image;
extern crate nalgebra;

use image::RgbImage;
use nalgebra::DMatrix;

use colormap;
use interop;
use inverse_depth::{self, InverseDepth};

pub type Float = f32;

// Create an RGB image of an inverse depth map.
pub fn idepth_image(idepth_map: &DMatrix<InverseDepth>) -> RgbImage {
    let viridis = &colormap::viridis_u8()[0..256];
    let (d_min, d_max) = min_max(idepth_map).unwrap();
    interop::rgb_from_matrix(
        &idepth_map.map(|idepth| idepth_enum_colormap(viridis, d_min, d_max, &idepth)),
    )
}

fn min_max(idepth_map: &DMatrix<InverseDepth>) -> Option<(Float, Float)> {
    let mut min_temp: Option<Float> = None;
    let mut max_temp: Option<Float> = None;
    idepth_map.iter().for_each(|idepth| {
        if let Some((d, _)) = inverse_depth::with_variance(idepth) {
            min_temp = min_temp.map(|x| x.min(d)).or(Some(d));
            max_temp = max_temp.map(|x| x.max(d)).or(Some(d));
        }
    });
    if let (Some(min_value), Some(max_value)) = (min_temp, max_temp) {
        Some((min_value, max_value))
    } else {
        None
    }
}

// INVERSE DEPTH HELPERS #############################################

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
pub fn idepth_enum_colormap(
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