// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Helper functions to handle datasets compatible with TUM RGB-D.

use nalgebra as na;
use std::path::PathBuf;

use crate::core::camera::Intrinsics;
use crate::misc::type_aliases::{Float, Iso3};

/// U16 depth values are scaled for better precision.
/// So 5000 in the 16 bits gray png corresponds to 1 meter.
pub const DEPTH_SCALE: Float = 5000.0;

/// Acceptable error is assimilated to 1cm at 1m.
/// The difference between 1/1m and 1/1.01m is ~ 0.01
/// so we will take a variance of 0.01^2 = 0.0001.
pub const VARIANCE_ICL_NUIM: Float = 0.0001;

/// Intrinsics parameters of the ICL-NUIM dataset.
pub const INTRINSICS_ICL_NUIM: Intrinsics = Intrinsics {
    principal_point: (319.5, 239.5),
    focal: (481.20, -480.00),
    skew: 0.0,
};

/// Intrinsics parameters of freiburg 1 (fr1) scenes in the TUM RGB-D dataset.
#[allow(clippy::excessive_precision)]
pub const INTRINSICS_FR1: Intrinsics = Intrinsics {
    principal_point: (318.643_040, 255.313_989),
    focal: (517.306_408, 516.469_215),
    skew: 0.0,
};

/// Intrinsics parameters of freiburg 2 (fr2) scenes in the TUM RGB-D dataset.
#[allow(clippy::excessive_precision)]
pub const INTRINSICS_FR2: Intrinsics = Intrinsics {
    principal_point: (325.141_442, 249.701_764),
    focal: (520.908_620, 521.007_327),
    skew: 0.0,
};

/// Intrinsics parameters of freiburg 3 (fr3) scenes in the TUM RGB-D dataset.
#[allow(clippy::excessive_precision)]
pub const INTRINSICS_FR3: Intrinsics = Intrinsics {
    principal_point: (320.106_653, 247.632_132),
    focal: (535.433_105, 539.212_524),
    skew: 0.0,
};

/// Timestamp and 3D camera pose of a frame.
#[derive(Debug)]
pub struct Frame {
    /// Timestamp of the frame.
    pub timestamp: f64,
    /// Pose (rigid body motion / direct isometry) of the frame.
    pub pose: Iso3,
}

/// Association of two related depth and color timestamps and images file paths.
#[derive(Debug)]
pub struct Association {
    /// Timestamp of the depth image.
    pub depth_timestamp: f64,
    /// File path of the depth image.
    pub depth_file_path: PathBuf,
    /// Timestamp of the color image.
    pub color_timestamp: f64,
    /// File path of the color image.
    pub color_file_path: PathBuf,
}

/// Write Frame data in the TUM RGB-D format for trajectories.
impl std::string::ToString for Frame {
    /// `timestamp tx ty tz qx qy qz qw`
    fn to_string(&self) -> String {
        let t = self.pose.translation.vector;
        let q = self.pose.rotation.into_inner().coords;
        format!(
            "{} {} {} {} {} {} {} {}",
            self.timestamp, t.x, t.y, t.z, q.x, q.y, q.z, q.w
        )
    }
}

/// Parse useful files (trajectories, associations, ...) in a dataset using the TUM RGB-D format.
pub mod parse {
    use super::*;
    use nom::{
        alt, anychar, do_parse, double, float, is_not, many0, map, named, space, tag,
        types::CompleteStr,
    };

    /// Parse an association file into a vector of `Association`.
    pub fn associations(file_content: &str) -> Result<Vec<Association>, String> {
        multi_line(association_line, file_content)
    }

    /// Parse a trajectory file into a vector of `Frame`.
    pub fn trajectory(file_content: &str) -> Result<Vec<Frame>, String> {
        multi_line(trajectory_line, file_content)
    }

    fn multi_line<F, T>(line_parser: F, file_content: &str) -> Result<Vec<T>, String>
    where
        F: Fn(CompleteStr) -> nom::IResult<CompleteStr, Option<T>>,
    {
        let mut vec_data = Vec::new();
        for line in file_content.lines() {
            match line_parser(CompleteStr(line)) {
                Ok((_, Some(data))) => vec_data.push(data),
                Ok(_) => (),
                Err(_) => return Err("Parsing error".to_string()),
            }
        }
        Ok(vec_data)
    }

    // nom parsers #############################################################

    // Associations --------------------

    // Association line is either a comment or two timestamps and file paths.
    named!(association_line<CompleteStr, Option<Association> >,
        alt!( map!(comment, |_| None) | map!(association, Some) )
    );

    // Parse an association of depth and color timestamps and file paths.
    named!(association<CompleteStr, Association>,
        do_parse!(
            depth_timestamp: double >> space >>
            depth_file_path: path >> space >>
            color_timestamp: double >> space >>
            color_file_path: path >>
            (Association { depth_timestamp, depth_file_path, color_timestamp, color_file_path })
        )
    );

    named!(path<CompleteStr, PathBuf>,
        map!(is_not!(" \t\r\n"), |s| PathBuf::from(*s))
    );

    // Trajectory ----------------------

    // Trajectory line is either a comment or a frame timestamp and pose.
    named!(trajectory_line<CompleteStr, Option<Frame> >,
        alt!( map!(comment, |_| None) | map!(frame, Some) )
    );

    // Parse a comment.
    named!(comment<CompleteStr,()>,
        do_parse!( tag!("#") >> many0!(anychar) >> ())
    );

    // Parse a frame.
    named!(frame<CompleteStr, Frame>,
        do_parse!(
            t: double >> space >>
            p: pose >>
            (Frame { timestamp: t, pose: p })
        )
    );

    // Parse extrinsics camera parameters.
    named!(pose<CompleteStr, Iso3 >,
        do_parse!(
            t: translation >> space >>
            r: rotation >>
            (Iso3::from_parts(t, r))
        )
    );

    // Parse components of a translation.
    named!(translation<CompleteStr, na::Translation3<Float> >,
        do_parse!(
            x: float >> space >>
            y: float >> space >>
            z: float >>
            (na::Translation3::new(x, y, z))
        )
    );

    // Parse components of a unit quaternion describing the rotation.
    named!(rotation<CompleteStr, na::UnitQuaternion<Float> >,
        do_parse!(
            qx: float >> space >>
            qy: float >> space >>
            qz: float >> space >>
            qw: float >>
            (na::UnitQuaternion::from_quaternion(na::Quaternion::new(qw, qx, qy, qz)))
        )
    );

} // pub mod parse
