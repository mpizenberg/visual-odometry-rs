// Interesting reads
// * Sophus c++ library: https://github.com/strasdat/Sophus
// * Ethan Eade course on Lie Groups for 2D and 3D transformations:
//     * details: http://ethaneade.com/lie.pdf
//     * summary: http://ethaneade.com/lie_groups.pdf

use nalgebra::{Matrix3, Quaternion, UnitQuaternion, Vector3};
use std::f32::consts::PI;

use crate::misc::type_aliases::Float;

pub type Element = Vector3<Float>;

const EPSILON_TAYLOR_SERIES: Float = 1e-2;
const EPSILON_TAYLOR_SERIES_2: Float = EPSILON_TAYLOR_SERIES * EPSILON_TAYLOR_SERIES;
const _1_8: Float = 0.125;
const _1_48: Float = 1.0 / 48.0;

// Hat operator.
// Goes from so3 parameterization to so3 element (skew-symmetric matrix).
pub fn hat(w: Element) -> Matrix3<Float> {
    Matrix3::new(0.0, -w.z, w.y, w.z, 0.0, -w.x, -w.y, w.x, 0.0)
}

// Squared hat operator (hat_2(w) == hat(w) * hat(w)).
// PS: result is a symmetric matrix.
pub fn hat_2(w: Element) -> Matrix3<Float> {
    let w11 = w.x * w.x;
    let w12 = w.x * w.y;
    let w22 = w.y * w.y;
    let w23 = w.y * w.z;
    let w33 = w.z * w.z;
    let w13 = w.x * w.z;
    #[cfg_attr(rustfmt, rustfmt_skip)]
    Matrix3::new(
        -w22 - w33,     w12,           w13,
         w12,          -w11 - w33,     w23,
         w13,           w23,          -w11 - w22,
    )
}

// Vee operator. Inverse of hat operator.
// Warning! does not check that the given matrix is skew-symmetric.
pub fn vee(mat: Matrix3<Float>) -> Element {
    Vector3::new(mat.m32, mat.m13, mat.m21)
}

// Compute the exponential map from Lie algebra so3 to Lie group SO3.
// Goes from so3 parameterization to SO3 element (rotation).
pub fn exp(w: Element) -> UnitQuaternion<Float> {
    let theta_2 = w.norm_squared();
    let real_factor;
    let imag_factor;
    if theta_2 < EPSILON_TAYLOR_SERIES_2 {
        real_factor = 1.0 - _1_8 * theta_2;
        imag_factor = 0.5 - _1_48 * theta_2;
    } else {
        let theta = theta_2.sqrt();
        let half_theta = 0.5 * theta;
        real_factor = half_theta.cos();
        imag_factor = half_theta.sin() / theta;
    }
    // This is actually already a unit quaternion so we should not use
    // the from_quaternion function that performs a renormalization.
    UnitQuaternion::from_quaternion(Quaternion::from_parts(real_factor, imag_factor * w))
}

// Compute the logarithm map from the Lie group SO3 to the Lie algebra so3.
// Inverse of the exponential map.
pub fn log(rotation: UnitQuaternion<Float>) -> Element {
    let imag_vector = rotation.vector();
    let imag_norm_2 = imag_vector.norm_squared();
    let real_factor = rotation.scalar();
    if imag_norm_2 < EPSILON_TAYLOR_SERIES_2 {
        let theta_by_imag_norm = 2.0 / real_factor; // TAYLOR
        theta_by_imag_norm * imag_vector
    } else if real_factor.abs() < EPSILON_TAYLOR_SERIES {
        let imag_norm = imag_norm_2.sqrt();
        let alpha = real_factor.abs() / imag_norm;
        let theta = real_factor.signum() * (PI - 2.0 * alpha); // TAYLOR
        (theta / imag_norm) * imag_vector
    } else {
        let imag_norm = imag_norm_2.sqrt();
        // Is atan correct? should I use atan2 instead?
        let theta = 2.0 * (imag_norm / real_factor).atan();
        (theta / imag_norm) * imag_vector
    }
}

// TESTS #############################################################

#[cfg(test)]
mod tests {

    use super::*;

    // The best precision I get for round trips with quickcheck random inputs
    // with exact trigonometric computations ("else" branches) is around 1e-6.
    const EPSILON_ROUNDTRIP_APPROX: Float = 1e-6;

    #[test]
    fn exp_log_round_trip() {
        let w = Vector3::zeros();
        assert_eq!(w, log(exp(w)));
    }

    // PROPERTY TESTS ################################################

    quickcheck! {
        fn hat_vee_roundtrip(x: Float, y: Float, z: Float) -> bool {
            let element = Vector3::new(x,y,z);
            element == vee(hat(element))
        }

        fn hat_2_ok(x: Float, y: Float, z: Float) -> bool {
            let element = Vector3::new(x,y,z);
            hat_2(element) == hat(element) * hat(element)
        }

        fn log_exp_round_trip(roll: Float, pitch: Float, yaw: Float) -> bool {
            let rotation = gen_rotation(roll, pitch, yaw);
            relative_eq!(
                rotation,
                exp(log(rotation)),
                epsilon = EPSILON_ROUNDTRIP_APPROX
            )
        }
    }

    // GENERATORS ####################################################

    fn gen_rotation(roll: Float, pitch: Float, yaw: Float) -> UnitQuaternion<Float> {
        UnitQuaternion::from_euler_angles(roll, pitch, yaw)
    }
}
