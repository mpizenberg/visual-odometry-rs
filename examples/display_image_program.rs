//! Display an image with conrod

#[macro_use]
extern crate conrod;
extern crate image;

mod program;

use conrod::backend::glium::glium;
use conrod::backend::glium::glium::Surface; // trait
use conrod::{color, widget, Colorable, Positionable, Sizeable, Widget};

const WIDTH: u32 = 620;
const HEIGHT: u32 = 480;

fn main() {
    // Init the program
    let mut prog = program::Program::new(WIDTH, HEIGHT, "Conrod image example");

    // The `WidgetId` for our background and `Image` widgets.
    widget_ids!(struct Ids { background, texture });
    let ids = Ids::new(prog.ui.widget_id_generator());

    // Create our `conrod::image::Map` which describes each of our widget->image mappings.
    // In our case we only have one image, however the macro may be used to list multiple.
    let raw_image = load_raw_image("data/images/0001.png");
    let texture = glium::texture::Texture2d::new(&prog.display, raw_image).unwrap();
    let (w, h) = (texture.width(), texture.height());
    let mut image_map = conrod::image::Map::new();
    let texture = image_map.insert(texture);

    // Poll events from the window.
    let mut event_loop = program::EventLoop::new(std::time::Duration::from_millis(16));
    'main: loop {
        // Handle all events.
        for event in event_loop.next(&mut prog.glium_events_loop) {
            // Use the `winit` backend to convert the winit event to a conrod one.
            if let Some(ev) = conrod::backend::winit::convert_event(event.clone(), &prog.display) {
                prog.ui.handle_event(ev);
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
        prog.draw(&mut |ui| {
            // Draw a light blue background.
            widget::Canvas::new()
                .color(color::LIGHT_BLUE)
                .set(ids.background, ui);
            // Instantiate the `Image` at its full size in the middle of the window.
            widget::Image::new(texture)
                .w_h(w as f64, h as f64)
                .middle()
                .set(ids.texture, ui);
        });

        // Render the `Ui` and then display it on the screen.
        if let Some(primitives) = prog.ui.draw_if_changed() {
            prog.renderer.fill(&prog.display, primitives, &image_map);
            let mut target = prog.display.draw();
            target.clear_color(0.0, 0.0, 0.0, 1.0);
            prog.renderer
                .draw(&prog.display, &mut target, &image_map)
                .unwrap();
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
