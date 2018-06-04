//! Display an image with conrod

#[macro_use]
extern crate conrod;
extern crate image;

mod support;

use conrod::backend::glium::glium;
use conrod::backend::glium::glium::Surface; // trait
use conrod::{color, widget, Colorable, Positionable, Sizeable, Widget};

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

fn main() {
    // Build the window.
    let mut events_loop = glium::glutin::EventsLoop::new();
    let window = glium::glutin::WindowBuilder::new()
        .with_title("Image Widget Demonstration")
        .with_dimensions(WIDTH, HEIGHT);
    let context = glium::glutin::ContextBuilder::new()
        .with_vsync(true)
        .with_multisampling(4);
    let display = glium::Display::new(window, context, &events_loop).unwrap();

    // construct our `Ui`.
    let mut ui = conrod::UiBuilder::new([WIDTH as f64, HEIGHT as f64]).build();

    // A type used for converting `conrod::render::Primitives` into `Command`s that can be used
    // for drawing to the glium `Surface`.
    let mut renderer = conrod::backend::glium::Renderer::new(&display).unwrap();

    // The `WidgetId` for our background and `Image` widgets.
    widget_ids!(struct Ids { background, texture });
    let ids = Ids::new(ui.widget_id_generator());

    // Create our `conrod::image::Map` which describes each of our widget->image mappings.
    // In our case we only have one image, however the macro may be used to list multiple.
    let raw_image = load_raw_image("data/images/0001.png");
    let texture = glium::texture::Texture2d::new(&display, raw_image).unwrap();
    let (w, h) = (texture.get_width(), texture.get_height().unwrap());
    let mut image_map = conrod::image::Map::new();
    let texture = image_map.insert(texture);

    // Poll events from the window.
    let mut event_loop = support::EventLoop::new();
    'main: loop {
        // Handle all events.
        for event in event_loop.next(&mut events_loop) {
            // Use the `winit` backend feature to convert the winit event to a conrod one.
            if let Some(event) = conrod::backend::winit::convert_event(event.clone(), &display) {
                ui.handle_event(event);
            }

            match event {
                glium::glutin::Event::WindowEvent { event, .. } => match event {
                    glium::glutin::WindowEvent::Closed => break 'main,
                    _ => (),
                },
                _ => (),
            }
        }

        // Instantiate the widgets.
        {
            let ui = &mut ui.set_widgets();
            // Draw a light blue background.
            widget::Canvas::new()
                .color(color::LIGHT_BLUE)
                .set(ids.background, ui);
            // Instantiate the `Image` at its full size in the middle of the window.
            widget::Image::new(texture)
                .w_h(w as f64, h as f64)
                .middle()
                .set(ids.texture, ui);
        }

        // Render the `Ui` and then display it on the screen.
        if let Some(primitives) = ui.draw_if_changed() {
            renderer.fill(&display, primitives, &image_map);
            let mut target = display.draw();
            target.clear_color(0.0, 0.0, 0.0, 1.0);
            renderer.draw(&display, &mut target, &image_map).unwrap();
            target.finish().unwrap();
        }
    }
}

// Function loading an image from a path
fn load_raw_image(path: &str) -> glium::texture::RawImage2d<u8> {
    let img_path = std::path::Path::new(path);
    let img_rgba = image::open(&img_path).expect("Cannot open image").to_rgba();
    let img_size = img_rgba.dimensions();
    glium::texture::RawImage2d::from_raw_rgba_reversed(&img_rgba.into_raw(), img_size)
}
