use nalgebra::{Affine2, Matrix3, Point2, Point3, Quaternion, Translation3, UnitQuaternion, Vector3};
use std::fs::File;
use std::io;
use std::io::prelude::Read;

pub type Float = f32;

#[derive(PartialEq, Debug, Clone)]
pub struct Camera {
    pub intrinsics: Intrinsics,
    pub extrinsics: Extrinsics,
}

impl Camera {
    pub fn new(intrinsics: Intrinsics, extrinsics: Extrinsics) -> Camera {
        Camera {
            intrinsics,
            extrinsics,
        }
    }

    pub fn project(&self, point: Point3<Float>) -> Vector3<Float> {
        self.intrinsics.project(self.extrinsics.project(point))
    }

    pub fn back_project(&self, point: Point2<Float>, depth: Float) -> Point3<Float> {
        self.extrinsics
            .back_project(self.intrinsics.back_project(point, depth))
    }

    pub fn multi_res(self, n: usize) -> Vec<Camera> {
        self.intrinsics
            .clone()
            .multi_res(n)
            .into_iter()
            .map(|intrinsics| Self::new(intrinsics, self.extrinsics.clone()))
            .collect()
    }
}

// EXTRINSICS ##############################################

#[derive(PartialEq, Debug, Clone)]
pub struct Extrinsics {
    pub translation: Translation3<Float>,
    pub rotation: UnitQuaternion<Float>,
}

impl Extrinsics {
    pub fn new(translation: Translation3<Float>, rotation: UnitQuaternion<Float>) -> Extrinsics {
        Extrinsics {
            translation,
            rotation,
        }
    }

    pub fn project(&self, point: Point3<Float>) -> Point3<Float> {
        self.rotation.inverse() * (self.translation.inverse() * point)
    }

    pub fn back_project(&self, point: Point3<Float>) -> Point3<Float> {
        self.translation * (self.rotation * point)
    }

    // pub fn in_coordinates_of(&self, other: &Extrinsics) -> Extrinsics {
    //     Extrinsics {
    //         translation: (other.rotation.inverse()
    //             * (self.translation * other.translation.inverse()))
    //             .translation,
    //         rotation: other.rotation.inverse() * self.rotation,
    //     }
    // }

    pub fn read_from_tum_file(file_path: &str) -> Result<Vec<Extrinsics>, io::Error> {
        let mut file_content = String::new();
        File::open(file_path)?.read_to_string(&mut file_content)?;
        let mut extrinsics = Vec::new();
        for line in file_content.lines() {
            let values: Vec<Float> = line.split(' ').filter_map(|s| s.parse().ok()).collect();
            assert_eq!(8, values.len(), "There was an issue in parsing:\n{}", line);
            let translation = Translation3::new(values[1], values[2], values[3]);
            let rotation = UnitQuaternion::from_quaternion(Quaternion::new(
                values[4],
                values[5],
                values[6],
                values[7],
            ));
            extrinsics.push(Extrinsics::new(translation, rotation));
        }
        Ok(extrinsics)
    }
}

// INTRINSICS ##############################################

#[derive(PartialEq, Debug, Clone)]
pub struct Intrinsics {
    pub principal_point: (Float, Float),
    pub focal_length: Float,
    pub scaling: (Float, Float),
    pub skew: Float,
}

impl Intrinsics {
    pub fn matrix(&self) -> Affine2<Float> {
        Affine2::from_matrix_unchecked(Matrix3::new(
            self.focal_length * self.scaling.0,
            self.skew,
            self.principal_point.0,
            0.0,
            self.focal_length * self.scaling.1,
            self.principal_point.1,
            0.0,
            0.0,
            1.0,
        ))
    }

    pub fn multi_res(self, n: usize) -> Vec<Intrinsics> {
        let mut intrinsics = Vec::new();
        if n > 0 {
            intrinsics.push(self);
            for _ in 1..n {
                let new = { intrinsics.last().unwrap().half_res() };
                intrinsics.push(new);
            }
        }
        intrinsics
    }

    pub fn half_res(&self) -> Intrinsics {
        let (cx, cy) = self.principal_point;
        let (sx, sy) = self.scaling;
        Intrinsics {
            principal_point: ((cx + 0.5) / 2.0 - 0.5, (cy + 0.5) / 2.0 - 0.5),
            focal_length: self.focal_length,
            scaling: (sx / 2.0, sy / 2.0),
            skew: self.skew,
        }
    }

    pub fn project(&self, point: Point3<Float>) -> Vector3<Float> {
        Vector3::new(
            self.focal_length * self.scaling.0 * point[0] + self.skew * point[1]
                + self.principal_point.0 * point[2],
            self.focal_length * self.scaling.1 * point[1] + self.principal_point.1 * point[2],
            point[2],
        )
    }

    pub fn back_project(&self, point: Point2<Float>, depth: Float) -> Point3<Float> {
        let z = depth;
        let y = (point[1] - self.principal_point.1) * z / (self.focal_length * self.scaling.1);
        let x = ((point[0] - self.principal_point.0) * z - self.skew * y)
            / (self.focal_length * self.scaling.0);
        Point3::new(x, y, z)
    }
}

// FUNCTIONS ###############################################
