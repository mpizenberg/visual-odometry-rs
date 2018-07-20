// Interesting reads
// * Sophus c++ library: https://github.com/strasdat/Sophus
// * Ethan Eade course on Lie Groups for 2D and 3D transformations:
//     * details: http://ethaneade.com/lie.pdf
//     * summary: http://ethaneade.com/lie_groups.pdf

use nalgebra::{Matrix3, Quaternion, UnitQuaternion, Vector3};
use std::f32::consts::PI;

pub type Float = f32;

const EPSILON_TAYLOR_SERIES: Float = 1e-2;
const _1_8: Float = 0.125;
const _1_48: Float = 1.0 / 48.0;

pub type Element = Vector3<Float>;

// Hat operator.
// Goes from so3 parameterization to so3 element (skew-symmetric matrix).
pub fn hat(w: Element) -> Matrix3<Float> {
    let w1 = w[0];
    let w2 = w[1];
    let w3 = w[2];
    Matrix3::from_column_slice(&[0.0, w3, -w2, -w3, 0.0, w1, w2, -w1, 0.0])
}

// Squared hat operator (hat_2(w) == hat(w) * hat(w)).
// PS: result is a symmetric matrix.
pub fn hat_2(w: Element) -> Matrix3<Float> {
    let w1 = w[0];
    let w2 = w[1];
    let w3 = w[2];
    let w11 = w1 * w1;
    let w12 = w1 * w2;
    let w22 = w2 * w2;
    let w23 = w2 * w3;
    let w33 = w3 * w3;
    let w13 = w1 * w3;
    Matrix3::from_column_slice(&[
        -w22 - w33,
        w12,
        w13,
        w12,
        -w11 - w33,
        w23,
        w13,
        w23,
        -w11 - w22,
    ])
}

// Vee operator. Inverse of hat operator.
// Warning! does not check that the given matrix is skew-symmetric.
pub fn vee(mat: Matrix3<Float>) -> Element {
    // TODO: improve performance.
    Vector3::from_column_slice(&[mat[(2, 1)], mat[(0, 2)], mat[(1, 0)]])
}

// Compute the exponential map from Lie algebra so3 to Lie group SO3.
// Goes from so3 parameterization to SO3 element (rotation).
// Also returns the norm of the Lie algebra element.
pub fn exp(w: Element) -> (UnitQuaternion<Float>, Float) {
    let theta_2 = w.norm_squared();
    let theta = theta_2.sqrt();
    let real_factor;
    let imag_factor;
    if theta < EPSILON_TAYLOR_SERIES {
        real_factor = 1.0 - _1_8 * theta_2;
        imag_factor = 0.5 - _1_48 * theta_2;
    } else {
        let half_theta = 0.5 * theta;
        real_factor = half_theta.cos();
        imag_factor = half_theta.sin() / theta;
    }
    let rotation =
        UnitQuaternion::from_quaternion(Quaternion::from_parts(real_factor, imag_factor * w));
    (rotation, theta)
}

// Compute the logarithm map from the Lie group SO3 to the Lie algebra so3.
// Inverse of the exponential map.
// Also returns the norm of the Lie algebra element.
pub fn log(rotation: UnitQuaternion<Float>) -> (Element, Float) {
    let imag_vector = rotation.vector();
    let imag_norm_2 = imag_vector.norm_squared();
    let imag_norm = imag_norm_2.sqrt();
    let real_factor = rotation.scalar();
    let theta;
    let tangent;
    if imag_norm < EPSILON_TAYLOR_SERIES {
        let atan_coef_by_imag_norm = 2.0 / real_factor; // TAYLOR
        theta = atan_coef_by_imag_norm * imag_norm;
        tangent = atan_coef_by_imag_norm * imag_vector;
    } else if real_factor.abs() < EPSILON_TAYLOR_SERIES {
        let alpha = real_factor.abs() / imag_norm;
        theta = real_factor.signum() * (PI - 2.0 * alpha); // TAYLOR
        tangent = (theta / imag_norm) * imag_vector;
    } else {
        theta = 2.0 * (imag_norm / real_factor).atan();
        tangent = (theta / imag_norm) * imag_vector;
    }
    (tangent, theta)
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
        assert_eq!(w, round_trip_from_algebra(w));
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
                round_trip_from_group(rotation),
                epsilon = EPSILON_ROUNDTRIP_APPROX
            )
        }
    }

    // GENERATORS ####################################################

    fn gen_rotation(roll: Float, pitch: Float, yaw: Float) -> UnitQuaternion<Float> {
        UnitQuaternion::from_euler_angles(roll, pitch, yaw)
    }

    // HELPERS #######################################################

    fn round_trip_from_algebra(w: Element) -> Element {
        let (rotation, _) = exp(w);
        let (new_w, _) = log(rotation);
        new_w
    }

    fn round_trip_from_group(rotation: UnitQuaternion<Float>) -> UnitQuaternion<Float> {
        let (w, _) = log(rotation);
        let (new_rotation, _) = exp(w);
        new_rotation
    }
}
