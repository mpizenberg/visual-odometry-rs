// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Helper functions to manipulate inverse depth data from depth images.

use crate::misc::type_aliases::Float;

/// An inverse depth can be one of three values: unknown, discarded, or known with a given
/// variance.
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum InverseDepth {
    /// Not known by the capture device.
    Unknown,
    /// Value was considered too unreliable and discarded.
    Discarded,
    /// `WithVariance(inverse_depth, variance)`: known but with a given uncertainty.
    WithVariance(Float, Float),
}

/// Transform a depth value from a depth map into an inverse depth value with a given scaling.
///
/// A value of 0 means that it is unknown.
pub fn from_depth(scale: Float, depth: u16, variance: Float) -> InverseDepth {
    match depth {
        0 => InverseDepth::Unknown,
        _ => InverseDepth::WithVariance(scale / Float::from(depth), variance),
    }
}

/// Transform inverse depth value back into a depth value with a given scaling.
///
/// Unknown or discarded values are encoded with 0.
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn to_depth(scale: Float, idepth: InverseDepth) -> u16 {
    match idepth {
        InverseDepth::WithVariance(x, _) => (scale / x).round() as u16,
        _ => 0,
    }
}

// Merging strategies ######################################

/// Fuse 4 inverse depth pixels of a bloc with a given merging strategy.
///
/// It only keeps the known values and passes them as a `Vec` into the given strategy.
pub fn fuse<F>(
    a: InverseDepth,
    b: InverseDepth,
    c: InverseDepth,
    d: InverseDepth,
    strategy: F,
) -> InverseDepth
where
    F: Fn(&[(Float, Float)]) -> InverseDepth,
{
    strategy(
        [a, b, c, d]
            .iter()
            .filter_map(with_variance)
            .collect::<Vec<_>>()
            .as_slice(),
    )
}

fn with_variance(idepth: &InverseDepth) -> Option<(Float, Float)> {
    if let InverseDepth::WithVariance(idepth, var) = *idepth {
        Some((idepth, var))
    } else {
        None
    }
}

/// Merge idepth pixels of a bloc into their mean idepth.
///
/// Just like in DSO, inverse depth do not have statistical variance
/// but some kind of "weight", proportional to how "trusty" they are.
/// So here, the variance is to be considered as a weight instead.
pub fn strategy_dso_mean(valid_values: &[(Float, Float)]) -> InverseDepth {
    match valid_values {
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

/// Only merge inverse depths that are statistically similar.
/// Others are discarded.
///
/// Variance increases if only one inverse depth is known.
/// Variance decreases if the four inverse depths are known.
pub fn strategy_statistically_similar(valid_values: &[(Float, Float)]) -> InverseDepth {
    match valid_values {
        [(d1, v1)] => InverseDepth::WithVariance(*d1, 2.0 * v1), // v = 2/1 * mean
        [(d1, v1), (d2, v2)] => {
            let new_d = (d1 * v2 + d2 * v1) / (v1 + v2);
            let new_v = (v1 + v2) / 2.0; // v = 2/2 * mean
            if (d1 - new_d).powi(2) < new_v && (d2 - new_d).powi(2) < new_v {
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
            if (d1 - new_d).powi(2) < new_v
                && (d2 - new_d).powi(2) < new_v
                && (d3 - new_d).powi(2) < new_v
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
            if (d1 - new_d).powi(2) < new_v
                && (d2 - new_d).powi(2) < new_v
                && (d3 - new_d).powi(2) < new_v
                && (d4 - new_d).powi(2) < new_v
            {
                InverseDepth::WithVariance(new_d, new_v)
            } else {
                InverseDepth::Discarded
            }
        }
        _ => InverseDepth::Unknown,
    }
}
