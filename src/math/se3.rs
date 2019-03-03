//! Lie algebra/group functions for 3D rigid body motion.
//!
//! Interesting reads:
//! - Sophus c++ library: https://github.com/strasdat/Sophus
//! - Ethan Eade course on Lie Groups for 2D and 3D transformations:
//!     - details: http://ethaneade.com/lie.pdf
//!     - summary: http://ethaneade.com/lie_groups.pdf

use nalgebra::{Quaternion, Translation3, UnitQuaternion};
use std::f32::consts::PI;
use std::ops::Add;

use crate::math::so3;
use crate::misc::type_aliases::{Float, Iso3, Mat3, Mat4, Vec3, Vec6};

const EPSILON_TAYLOR_SERIES: Float = 1e-2;
const EPSILON_TAYLOR_SERIES_2: Float = EPSILON_TAYLOR_SERIES * EPSILON_TAYLOR_SERIES;
const _1_6: Float = 1.0 / 6.0;
const _1_8: Float = 0.125;
const _1_12: Float = 1.0 / 12.0;
const _1_15: Float = 1.0 / 15.0;
const _1_24: Float = 1.0 / 24.0;
const _1_48: Float = 1.0 / 48.0;
const _1_120: Float = 1.0 / 120.0;

// TODO: change to Vec6.
/// Parameterization of a twist (element of se3).
#[derive(PartialEq, Debug, Copy, Clone)]
pub struct Twist {
    v: Vec3,
    w: Vec3,
}

impl Add for Twist {
    type Output = Twist;
    fn add(self, other: Twist) -> Twist {
        Twist {
            v: self.v + other.v,
            w: self.w + other.w,
        }
    }
}

pub fn from_vector(vec: Vec6) -> Twist {
    Twist {
        v: Vec3::new(vec[0], vec[1], vec[2]),
        w: Vec3::new(vec[3], vec[4], vec[5]),
    }
}

pub fn to_vector(twist: Twist) -> Vec6 {
    let v = twist.v;
    let w = twist.w;
    Vec6::new(v[0], v[1], v[2], w[0], w[1], w[2])
}

// Hat operator. Goes from se3 parameters to se3 element (4x4 matrix).
pub fn hat(xi: Twist) -> Mat4 {
    let w1 = xi.w[0];
    let w2 = xi.w[1];
    let w3 = xi.w[2];
    let v1 = xi.v[0];
    let v2 = xi.v[1];
    let v3 = xi.v[2];
    #[cfg_attr(rustfmt, rustfmt_skip)]
    Mat4::new(
         0.0,  -w3,    w2,   v1,
         w3,    0.0,  -w1,   v2,
        -w2,    w1,    0.0,  v3,
         0.0,   0.0,   0.0,  0.0,
    )
}

// Vee operator. Inverse of hat operator.
// Warning! does not check that the given top left 3x3 sub-matrix is skew-symmetric.
pub fn vee(mat: Mat4) -> Twist {
    Twist {
        v: Vec3::new(mat.m14, mat.m24, mat.m34),
        w: Vec3::new(mat.m32, mat.m13, mat.m21),
    }
}

// Compute the exponential map from Lie algebra se3 to Lie group SE3.
// Goes from se3 parameterization to SE3 element (rigid body motion).
pub fn exp(xi: Twist) -> Iso3 {
    let theta_2 = xi.w.norm_squared();
    let (omega, omega_2) = (so3::hat(xi.w), so3::hat_2(xi.w));
    if theta_2 < EPSILON_TAYLOR_SERIES_2 {
        let real_factor = 1.0 - _1_8 * theta_2; // TAYLOR
        let imag_factor = 0.5 - _1_48 * theta_2; // TAYLOR
        let coef_omega = 0.5 - _1_24 * theta_2; // TAYLOR
        let coef_omega_2 = _1_6 - _1_120 * theta_2; // TAYLOR
        let v = Mat3::identity() + coef_omega * omega + coef_omega_2 * omega_2;
        let rotation = UnitQuaternion::from_quaternion(Quaternion::from_parts(
            real_factor,
            imag_factor * xi.w,
        ));
        Iso3::from_parts(Translation3::from(v * xi.v), rotation)
    } else {
        let theta = theta_2.sqrt();
        let half_theta = 0.5 * theta;
        let real_factor = half_theta.cos();
        let imag_factor = half_theta.sin() / theta;
        let coef_omega = (1.0 - theta.cos()) / theta_2;
        let coef_omega_2 = (theta - theta.sin()) / (theta * theta_2);
        let v = Mat3::identity() + coef_omega * omega + coef_omega_2 * omega_2;
        let rotation = UnitQuaternion::from_quaternion(Quaternion::from_parts(
            real_factor,
            imag_factor * xi.w,
        ));
        Iso3::from_parts(Translation3::from(v * xi.v), rotation)
    }
}

// Compute the logarithm map from the Lie group SE3 to the Lie algebra se3.
// Inverse of the exponential map.
pub fn log(iso: Iso3) -> Twist {
    let imag_vector = iso.rotation.vector();
    let imag_norm_2 = imag_vector.norm_squared();
    let real_factor = iso.rotation.scalar();
    if imag_norm_2 < EPSILON_TAYLOR_SERIES_2 {
        let theta_by_imag_norm = 2.0 / real_factor; // TAYLOR
        let w = theta_by_imag_norm * imag_vector;
        let (omega, omega_2) = (so3::hat(w), so3::hat_2(w));
        let x_2 = imag_norm_2 / (real_factor * real_factor);
        let coef_omega_2 = _1_12 * (1.0 + _1_15 * x_2); // TAYLOR
        let v_inv = Mat3::identity() - 0.5 * omega + coef_omega_2 * omega_2;
        Twist {
            v: v_inv * iso.translation.vector,
            w: w,
        }
    } else {
        let imag_norm = imag_norm_2.sqrt();
        let theta = if real_factor.abs() < EPSILON_TAYLOR_SERIES {
            let alpha = real_factor.abs() / imag_norm;
            real_factor.signum() * (PI - 2.0 * alpha) // TAYLOR
        } else {
            // Is this correct? should I use atan2 instead?
            2.0 * (imag_norm / real_factor).atan()
        };
        let theta_2 = theta * theta;
        let w = (theta / imag_norm) * imag_vector;
        let (omega, omega_2) = (so3::hat(w), so3::hat_2(w));
        let coef_omega_2 = (1.0 - 0.5 * theta * real_factor / imag_norm) / theta_2;
        let v_inv = Mat3::identity() - 0.5 * omega + coef_omega_2 * omega_2;
        Twist {
            v: v_inv * iso.translation.vector,
            w: w,
        }
    }
}

// TESTS #############################################################

#[cfg(test)]
mod tests {

    use super::*;

    // The best precision I get for round trips with quickcheck random inputs
    // with exact trigonometric computations ("else" branches) is around 1e-4.
    const EPSILON_ROUNDTRIP_APPROX: Float = 1e-4;

    #[test]
    fn exp_log_round_trip() {
        let v = Vec3::zeros();
        let w = Vec3::zeros();
        let xi = Twist { v, w };
        assert_eq!(xi, log(exp(xi)));
    }

    // PROPERTY TESTS ################################################

    quickcheck! {
        fn hat_vee_roundtrip(v1: Float, v2: Float, v3: Float, w1: Float, w2: Float, w3: Float) -> bool {
            let v = Vec3::new(v1, v2, v3);
            let w = Vec3::new(w1, w2, w3);
            let twist = Twist { v, w };
            twist == vee(hat(twist))
        }

        fn log_exp_round_trip(t1: Float, t2: Float, t3:Float, a1: Float, a2: Float, a3: Float) -> bool {
            let rigid_motion = gen_rigid_motion(&[t1,t2,t3], &[a1,a2,a3]);
            relative_eq!(
                rigid_motion,
                exp(log(rigid_motion)),
                epsilon = EPSILON_ROUNDTRIP_APPROX
            )
        }
    }

    // GENERATORS ####################################################

    fn gen_rigid_motion(translation_slice: &[Float; 3], angles: &[Float; 3]) -> Iso3 {
        let translation = Translation3::from(Vec3::from_column_slice(translation_slice));
        let rotation = UnitQuaternion::from_euler_angles(angles[0], angles[1], angles[2]);
        Iso3::from_parts(translation, rotation)
    }
}
