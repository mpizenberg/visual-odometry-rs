type Float = f32;

#[derive(Serialize, Deserialize, PartialEq, Debug)]
pub enum Intrinsics {
    Pinhole(Pinhole),
    PinholeWithDistortion(Pinhole, RadialTangential),
    FishEye(FishEye),
    Fov(Fov),
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
pub struct Pinhole {
    pub principal_point: (Float, Float),
    pub focal_length: Float,
    pub scaling: (Float, Float),
    pub skew: Float,
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
pub struct RadialTangential {
    pub radial_coeffs: (Float, Float, Float),
    pub tangential_coeffs: (Float, Float),
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
pub struct FishEye {}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
pub struct Fov {}
