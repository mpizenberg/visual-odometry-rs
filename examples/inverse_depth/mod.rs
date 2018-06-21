extern crate rand;

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum InverseDepth {
    Unknown,
    Discarded,
    WithVariance(f32, f32),
}

// Transform depth value from dataset into an inverse depth value.
// STD is assimilated to 10cm so variance = ( 1 / 0.10 ) ^ 2 = 100.
pub fn from_depth(depth: u16) -> InverseDepth {
    match depth {
        0 => InverseDepth::Unknown,
        _ => InverseDepth::WithVariance(5000f32 / depth as f32, 100f32),
    }
}

// Visualize as an 8-bits intensity.
pub fn visual_enum(idepth: &InverseDepth) -> u8 {
    match idepth {
        InverseDepth::Unknown => 0u8,
        InverseDepth::Discarded => 50u8,
        InverseDepth::WithVariance(_, _) => 255u8,
    }
}

// Fuse 4 sub pixels with inverse depths.
pub fn fuse(a: InverseDepth, b: InverseDepth, c: InverseDepth, d: InverseDepth) -> InverseDepth {
    let those_with_variance: Vec<(f32, f32)> =
        [a, b, c, d].iter().filter_map(with_variance).collect();
    if those_with_variance.len() == 0 {
        InverseDepth::Unknown
    } else {
        let (idepth, var) = those_with_variance.last().unwrap();
        if rand::random() {
            InverseDepth::WithVariance(*idepth, *var)
        } else {
            InverseDepth::Discarded
        }
    }
}

pub fn with_variance(idepth: &InverseDepth) -> Option<(f32, f32)> {
    if let InverseDepth::WithVariance(idepth, var) = idepth {
        Some((*idepth, *var))
    } else {
        None
    }
}
