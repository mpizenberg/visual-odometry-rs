// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use image;
use nalgebra::{DMatrix, Matrix4, Point2, Point3};
use std::{env, error::Error, fs, io::BufReader, io::Read, path::Path, path::PathBuf};

use visual_odometry_rs as vors;
use vors::core::camera::Intrinsics;
use vors::core::track::inverse_compositional::{self as track, Tracker};
use vors::dataset::tum_rgbd;
use vors::misc::{helper, interop};

use kiss3d::camera::Camera;
use kiss3d::context::Context;
use kiss3d::planar_camera::PlanarCamera;
use kiss3d::post_processing::PostProcessingEffect;
use kiss3d::renderer::Renderer;
use kiss3d::resource::{
    AllocationType, BufferType, Effect, GPUVec, ShaderAttribute, ShaderUniform,
};
use kiss3d::text::Font;
use kiss3d::window::{State, Window};

fn main() {
    let args: Vec<String> = env::args().collect();
    if let Err(error) = my_run(&args) {
        eprintln!("{:?}", error);
    }
}

const USAGE: &str = "Usage: ./vors_track [fr1|fr2|fr3|icl] associations_file";

fn my_run(args: &[String]) -> Result<(), Box<Error>> {
    // Init tracker and associations.
    let (tracker, associations) = init(args)?;

    // Init kiss3d stuff.
    let window = Window::new("Kiss3d: persistent_point_cloud");
    let app = AppState::new(tracker, associations);

    // Read one frame per render frame
    Ok(window.render_loop(app))
}

fn init(args: &[String]) -> Result<(Tracker, Vec<tum_rgbd::Association>), Box<Error>> {
    // Check that the arguments are correct.
    let valid_args = check_args(args)?;

    // Build a vector containing timestamps and full paths of images.
    let associations = parse_associations(valid_args.associations_file_path)?;

    // Setup tracking configuration.
    let config = track::Config {
        nb_levels: 6,
        candidates_diff_threshold: 7,
        depth_scale: tum_rgbd::DEPTH_SCALE,
        intrinsics: valid_args.intrinsics,
        idepth_variance: 0.0001,
    };

    // Initialize tracker with first depth and color image.
    let (depth_map, img) = read_images(&associations[0])?;
    let depth_time = associations[0].depth_timestamp;
    let img_time = associations[0].color_timestamp;
    let tracker = config.init(depth_time, &depth_map, img_time, img);
    Ok((tracker, associations))
}

struct Args {
    associations_file_path: PathBuf,
    intrinsics: Intrinsics,
}

/// Verify that command line arguments are correct.
fn check_args(args: &[String]) -> Result<Args, String> {
    // eprintln!("{:?}", args);
    if let [_, camera_id, associations_file_path_str] = args {
        let intrinsics = create_camera(camera_id)?;
        let associations_file_path = PathBuf::from(associations_file_path_str);
        if associations_file_path.is_file() {
            Ok(Args {
                intrinsics,
                associations_file_path,
            })
        } else {
            eprintln!("{}", USAGE);
            Err(format!(
                "The association file does not exist or is not reachable: {}",
                associations_file_path_str
            ))
        }
    } else {
        eprintln!("{}", USAGE);
        Err("Wrong number of arguments".to_string())
    }
}

/// Create camera depending on `camera_id` command line argument.
fn create_camera(camera_id: &str) -> Result<Intrinsics, String> {
    match camera_id {
        "fr1" => Ok(tum_rgbd::INTRINSICS_FR1),
        "fr2" => Ok(tum_rgbd::INTRINSICS_FR2),
        "fr3" => Ok(tum_rgbd::INTRINSICS_FR3),
        "icl" => Ok(tum_rgbd::INTRINSICS_ICL_NUIM),
        _ => {
            eprintln!("{}", USAGE);
            Err(format!("Unknown camera id: {}", camera_id))
        }
    }
}

/// Open an association file and parse it into a vector of Association.
fn parse_associations<P: AsRef<Path>>(
    file_path: P,
) -> Result<Vec<tum_rgbd::Association>, Box<Error>> {
    let file = fs::File::open(&file_path)?;
    let mut file_reader = BufReader::new(file);
    let mut content = String::new();
    file_reader.read_to_string(&mut content)?;
    tum_rgbd::parse::associations(&content)
        .map(|v| v.iter().map(|a| abs_path(&file_path, a)).collect())
        .map_err(|s| s.into())
}

/// Transform relative images file paths into absolute ones.
fn abs_path<P: AsRef<Path>>(file_path: P, assoc: &tum_rgbd::Association) -> tum_rgbd::Association {
    let parent = file_path
        .as_ref()
        .parent()
        .expect("How can this have no parent");
    tum_rgbd::Association {
        depth_timestamp: assoc.depth_timestamp,
        depth_file_path: parent.join(&assoc.depth_file_path),
        color_timestamp: assoc.color_timestamp,
        color_file_path: parent.join(&assoc.color_file_path),
    }
}

/// Read a depth and color image given by an association.
fn read_images(assoc: &tum_rgbd::Association) -> Result<(DMatrix<u16>, DMatrix<u8>), Box<Error>> {
    let (w, h, depth_map_vec_u16) = helper::read_png_16bits(&assoc.depth_file_path)?;
    let depth_map = DMatrix::from_row_slice(h, w, depth_map_vec_u16.as_slice());
    let img = interop::matrix_from_image(image::open(&assoc.color_file_path)?.to_luma());
    Ok((depth_map, img))
}

// Kiss3d stuff ##########################################################################

// Custom renderers are used to allow rendering objects that are not necessarily
// represented as meshes. In this example, we will render a large, growing, point cloud
// with a color associated to each point.

// Writing a custom renderer requires the main loop to be
// handled by the `State` trait instead of a `while window.render()`
// like other examples.

struct AppState {
    tracker: Tracker,
    associations: Vec<tum_rgbd::Association>,
    next_frame: usize,
    point_cloud_renderer: PointCloudRenderer,
}

impl AppState {
    fn new(tracker: Tracker, associations: Vec<tum_rgbd::Association>) -> Self {
        let point_cloud_renderer = PointCloudRenderer::new(1.0);
        AppState {
            tracker,
            associations,
            next_frame: 0,
            point_cloud_renderer,
        }
    }
}

impl State for AppState {
    // Return the custom renderer that will be called at each
    // render loop.
    fn cameras_and_effect_and_renderer(
        &mut self,
    ) -> (
        Option<&mut dyn Camera>,
        Option<&mut dyn PlanarCamera>,
        Option<&mut dyn Renderer>,
        Option<&mut dyn PostProcessingEffect>,
    ) {
        (None, None, Some(&mut self.point_cloud_renderer), None)
    }

    fn step(&mut self, window: &mut Window) {
        // Track every frame in the associations file.
        if self.next_frame < self.associations.len() {
            let assoc = &self.associations[self.next_frame];
            self.next_frame += 1;

            // Load depth and color images.
            let (depth_map, img) = read_images(assoc).expect("read_images error");

            // Track the rgb-d image.
            let change_keyframe = self.tracker.track(
                assoc.depth_timestamp,
                &depth_map,
                assoc.color_timestamp,
                img,
            );

            // Print to stdout the frame pose.
            let (timestamp, pose) = self.tracker.current_frame();
            println!("{}", (tum_rgbd::Frame { timestamp, pose }).to_string());

            // Render 3d points.
            if change_keyframe {
                let color = rand::random();
                for point in self.tracker.points_3d().iter() {
                    self.point_cloud_renderer.push(*point, color);
                }
            }

            let num_points_text = format!(
                "Number of points: {}",
                self.point_cloud_renderer.num_points()
            );

            window.draw_text(
                &num_points_text,
                &Point2::new(0.0, 20.0),
                60.0,
                &Font::default(),
                &Point3::new(1.0, 1.0, 1.0),
            );
        }
    }
}

/// Structure which manages the display of long-living points.
struct PointCloudRenderer {
    shader: Effect,
    pos: ShaderAttribute<Point3<f32>>,
    color: ShaderAttribute<Point3<f32>>,
    proj: ShaderUniform<Matrix4<f32>>,
    view: ShaderUniform<Matrix4<f32>>,
    colored_points: GPUVec<Point3<f32>>,
    point_size: f32,
}

impl PointCloudRenderer {
    /// Creates a new points renderer.
    fn new(point_size: f32) -> PointCloudRenderer {
        let mut shader = Effect::new_from_str(VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC);

        shader.use_program();

        PointCloudRenderer {
            colored_points: GPUVec::new(Vec::new(), BufferType::Array, AllocationType::StreamDraw),
            pos: shader.get_attrib::<Point3<f32>>("position").unwrap(),
            color: shader.get_attrib::<Point3<f32>>("color").unwrap(),
            proj: shader.get_uniform::<Matrix4<f32>>("proj").unwrap(),
            view: shader.get_uniform::<Matrix4<f32>>("view").unwrap(),
            shader,
            point_size,
        }
    }

    fn push(&mut self, point: Point3<f32>, color: Point3<f32>) {
        if let Some(colored_points) = self.colored_points.data_mut() {
            colored_points.push(point);
            colored_points.push(color);
        }
    }

    fn num_points(&self) -> usize {
        self.colored_points.len() / 2
    }
}

impl Renderer for PointCloudRenderer {
    /// Actually draws the points.
    fn render(&mut self, pass: usize, camera: &mut dyn Camera) {
        if self.colored_points.len() == 0 {
            return;
        }

        self.shader.use_program();
        self.pos.enable();
        self.color.enable();

        camera.upload(pass, &mut self.proj, &mut self.view);

        self.color.bind_sub_buffer(&mut self.colored_points, 1, 1);
        self.pos.bind_sub_buffer(&mut self.colored_points, 1, 0);

        let ctxt = Context::get();
        ctxt.point_size(self.point_size);
        ctxt.draw_arrays(Context::POINTS, 0, (self.colored_points.len() / 2) as i32);

        self.pos.disable();
        self.color.disable();
    }
}

const VERTEX_SHADER_SRC: &'static str = "#version 100
    attribute vec3 position;
    attribute vec3 color;
    varying   vec3 Color;
    uniform   mat4 proj;
    uniform   mat4 view;
    void main() {
        gl_Position = proj * view * vec4(position, 1.0);
        Color = color;
    }";

const FRAGMENT_SHADER_SRC: &'static str = "#version 100
#ifdef GL_FRAGMENT_PRECISION_HIGH
   precision highp float;
#else
   precision mediump float;
#endif

    varying vec3 Color;
    void main() {
        gl_FragColor = vec4(Color, 1.0);
    }";
