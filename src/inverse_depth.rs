use rand;

pub type Float = f32;

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum InverseDepth {
    Unknown,
    Discarded,
    WithVariance(Float, Float),
}

// Acceptable error is assimilated to 1cm at 1m.
// The difference between 1/1m and 1/1.01m is ~ 0.01
// So we will take a variance of 0.01^2 = 0.0001
pub const DEFAULT_VARIANCE: Float = 0.0001;

// Transform depth value from dataset into an inverse depth value.
pub fn from_depth(scale: Float, depth: u16) -> InverseDepth {
    match depth {
        0 => InverseDepth::Unknown,
        _ => InverseDepth::WithVariance(scale / depth as Float, DEFAULT_VARIANCE),
    }
}

// Transform InverseDepth back into a depth
// with the same scaling as in the dataset.
pub fn to_depth(scale: Float, idepth: InverseDepth) -> u16 {
    match idepth {
        InverseDepth::WithVariance(x, _) => (scale / x).round() as u16,
        _ => 0,
    }
}

// Merging strategies ######################################

// Fuse 4 sub pixels with inverse depths.
pub fn fuse<F>(
    a: InverseDepth,
    b: InverseDepth,
    c: InverseDepth,
    d: InverseDepth,
    strategy: F,
) -> InverseDepth
where
    F: Fn(Vec<(Float, Float)>) -> InverseDepth,
{
    strategy(
        [a, b, c, d]
            .iter()
            .filter_map(with_variance)
            .collect::<Vec<_>>(),
    )
}

pub fn with_variance(idepth: &InverseDepth) -> Option<(Float, Float)> {
    if let InverseDepth::WithVariance(idepth, var) = idepth {
        Some((*idepth, *var))
    } else {
        None
    }
}

pub fn strategy_random(valid_values: Vec<(Float, Float)>) -> InverseDepth {
    match valid_values.as_slice() {
        [(idepth, var)] | [(idepth, var), _] | [(idepth, var), _, _] | [(idepth, var), _, _, _] => {
            if rand::random() {
                InverseDepth::WithVariance(*idepth, *var)
            } else {
                InverseDepth::Discarded
            }
        }
        _ => InverseDepth::Unknown,
    }
}

// In DSO, inverse depth do not have statistical variance
// but some kind of "weight", proportional to how "trusty" they are.
// So here, the variance is to be considered as a weight instead.
pub fn strategy_dso_mean(valid_values: Vec<(Float, Float)>) -> InverseDepth {
    match valid_values.as_slice() {
        [] => InverseDepth::Unknown,
        [(d1, v1)] => InverseDepth::WithVariance(*d1, *v1),
        [(d1, v1), (d2, v2)] => {
            let sum = v1 + v2;
            InverseDepth::WithVariance((d1 * v1 + d2 * v2) / sum, sum)
        }
        [(d1, v1), (d2, v2), (d3, v3)] => {
            let sum = v1 + v2 + v3;
            InverseDepth::WithVariance((d1 * v1 + d2 * v2 + d3 * v3) / sum, sum)
        }
        [(d1, v1), (d2, v2), (d3, v3), (d4, v4)] => {
            let sum = v1 + v2 + v3 + v4;
            InverseDepth::WithVariance((d1 * v1 + d2 * v2 + d3 * v3 + d4 * v4) / sum, sum)
        }
        _ => InverseDepth::Unknown,
    }
}

// Only merge inverse depths that are statistically similar.
pub fn strategy_statistically_similar(valid_values: Vec<(Float, Float)>) -> InverseDepth {
    match valid_values.as_slice() {
        [] => InverseDepth::Unknown,
        [(d1, v1)] => InverseDepth::WithVariance(*d1, 2.0 * v1), // v = 2/1 * mean
        [(d1, v1), (d2, v2)] => {
            let new_d = (d1 * v2 + d2 * v1) / (v1 + v2);
            let new_v = (v1 + v2) / 2.0; // v = 2/2 * mean
            let new_std = new_v.sqrt();
            if (d1 - new_d).abs() < new_std && (d2 - new_d).abs() < new_std {
                InverseDepth::WithVariance(new_d, new_v)
            } else {
                InverseDepth::Discarded
            }
        }
        [(d1, v1), (d2, v2), (d3, v3)] => {
            let v12 = v1 * v2;
            let v13 = v1 * v3;
            let v23 = v2 * v3;
            let new_d = (d1 * v23 + d2 * v13 + d3 * v12) / (v12 + v13 + v23);
            let new_v = 2.0 * (v1 + v2 + v3) / 9.0; // v = 2/3 * mean
            let new_std = new_v.sqrt();
            if (d1 - new_d).abs() < new_std
                && (d2 - new_d).abs() < new_std
                && (d3 - new_d).abs() < new_std
            {
                InverseDepth::WithVariance(new_d, new_v)
            } else {
                InverseDepth::Discarded
            }
        }
        [(d1, v1), (d2, v2), (d3, v3), (d4, v4)] => {
            let v123 = v1 * v2 * v3;
            let v234 = v2 * v3 * v4;
            let v341 = v3 * v4 * v1;
            let v412 = v4 * v1 * v2;
            let sum_v1234 = v123 + v234 + v341 + v412;
            let new_d = (d1 * v234 + d2 * v341 + d3 * v412 + d4 * v123) / sum_v1234;
            let new_v = (v1 + v2 + v3 + v4) / 8.0; // v = 2/4 * mean
            let new_std = new_v.sqrt();
            if (d1 - new_d).abs() < new_std
                && (d2 - new_d).abs() < new_std
                && (d3 - new_d).abs() < new_std
                && (d4 - new_d).abs() < new_std
            {
                InverseDepth::WithVariance(new_d, new_v)
            } else {
                InverseDepth::Discarded
            }
        }
        _ => InverseDepth::Unknown,
    }
}
