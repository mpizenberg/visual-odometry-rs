// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Lie algebra/group functions for 3D rigid body motion.
//!
//! Interesting reads:
//! - Sophus c++ library: <https://github.com/strasdat/Sophus>
//! - Ethan Eade course on Lie Groups for 2D and 3D transformations:
//!     - details: <http://ethaneade.com/lie.pdf>
//!     - summary: <http://ethaneade.com/lie_groups.pdf>

use nalgebra::{Quaternion, Translation3, UnitQuaternion};
use std::f32::consts::PI;

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

/// Parameterization of a twist (element of se3).
pub type Twist = Vec6;

/// Retrieve the linear velocity part of the twist parameterization.
pub fn linear_velocity(xi: Twist) -> Vec3 {
    Vec3::new(xi[0], xi[1], xi[2])
}

/// Retrieve the angular velocity part of the twist parameterization.
pub fn angular_velocity(xi: Twist) -> Vec3 {
    Vec3::new(xi[3], xi[4], xi[5])
}

/// Hat operator.
/// Goes from se3 parameters to se3 element (4x4 matrix).
#[rustfmt::skip]
pub fn hat(xi: Twist) -> Mat4 {
    let w1 = xi[3];
    let w2 = xi[4];
    let w3 = xi[5];
    Mat4::new(
         0.0,  -w3,    w2,   xi[0],
         w3,    0.0,  -w1,   xi[1],
        -w2,    w1,    0.0,  xi[2],
         0.0,   0.0,   0.0,  0.0,
    )
}

/// Vee operator. Inverse of hat operator.
/// Warning! does not check that the given top left 3x3 sub-matrix is skew-symmetric.
pub fn vee(mat: Mat4) -> Twist {
    Vec6::new(mat.m14, mat.m24, mat.m34, mat.m32, mat.m13, mat.m21)
}

/// Compute the exponential map from Lie algebra se3 to Lie group SE3.
/// Goes from se3 parameterization to SE3 element (rigid body motion).
pub fn exp(xi: Twist) -> Iso3 {
    let xi_v = linear_velocity(xi);
    let xi_w = angular_velocity(xi);
    let theta_2 = xi_w.norm_squared();
    let (omega, omega_2) = (so3::hat(xi_w), so3::hat_2(xi_w));
    if theta_2 < EPSILON_TAYLOR_SERIES_2 {
        let real_factor = 1.0 - _1_8 * theta_2; // TAYLOR
        let imag_factor = 0.5 - _1_48 * theta_2; // TAYLOR
        let coef_omega = 0.5 - _1_24 * theta_2; // TAYLOR
        let coef_omega_2 = _1_6 - _1_120 * theta_2; // TAYLOR
        let v = Mat3::identity() + coef_omega * omega + coef_omega_2 * omega_2;
        let rotation = UnitQuaternion::from_quaternion(Quaternion::from_parts(
            real_factor,
            imag_factor * xi_w,
        ));
        Iso3::from_parts(Translation3::from(v * xi_v), rotation)
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
            imag_factor * xi_w,
        ));
        Iso3::from_parts(Translation3::from(v * xi_v), rotation)
    }
}

/// Compute the logarithm map from the Lie group SE3 to the Lie algebra se3.
/// Inverse of the exponential map.
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
        let xi_v = v_inv * iso.translation.vector;
        Vec6::new(xi_v[0], xi_v[1], xi_v[2], w[0], w[1], w[2])
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
        let xi_v = v_inv * iso.translation.vector;
        Vec6::new(xi_v[0], xi_v[1], xi_v[2], w[0], w[1], w[2])
    }
}

// TESTS #############################################################

#[cfg(test)]
mod tests {

    use super::*;
    use approx;
    use quickcheck_macros;

    // The best precision I get for round trips with quickcheck random inputs
    // with exact trigonometric computations ("else" branches) is around 1e-4.
    const EPSILON_ROUNDTRIP_APPROX: Float = 1e-4;

    #[test]
    fn exp_log_round_trip() {
        let xi = Vec6::zeros();
        assert_eq!(xi, log(exp(xi)));
    }

    // PROPERTY TESTS ################################################

    #[quickcheck_macros::quickcheck]
    fn hat_vee_roundtrip(v1: Float, v2: Float, v3: Float, w1: Float, w2: Float, w3: Float) -> bool {
        let xi = Vec6::new(v1, v2, v3, w1, w2, w3);
        xi == vee(hat(xi))
    }

    #[quickcheck_macros::quickcheck]
    fn log_exp_round_trip(
        t1: Float,
        t2: Float,
        t3: Float,
        a1: Float,
        a2: Float,
        a3: Float,
    ) -> bool {
        let rigid_motion = gen_rigid_motion(t1, t2, t3, a1, a2, a3);
        approx::relative_eq!(
            rigid_motion,
            exp(log(rigid_motion)),
            epsilon = EPSILON_ROUNDTRIP_APPROX
        )
    }

    // GENERATORS ####################################################

    fn gen_rigid_motion(t1: Float, t2: Float, t3: Float, a1: Float, a2: Float, a3: Float) -> Iso3 {
        let translation = Translation3::from(Vec3::new(t1, t2, t3));
        let rotation = UnitQuaternion::from_euler_angles(a1, a2, a3);
        Iso3::from_parts(translation, rotation)
    }
}
