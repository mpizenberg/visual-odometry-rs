extern crate conrod;

use conrod::backend::glium::glium;
use conrod::backend::glium::glium::Surface; // trait
use std;

pub struct Program {
    pub event_loop: EventLoop,
    pub ui: conrod::Ui,
    // stuff from glium
    pub glium_events_loop: glium::glutin::EventsLoop,
    pub display: glium::Display,
    pub renderer: conrod::backend::glium::Renderer,
}

pub enum Continuation {
    Stop,
    Continue,
}

impl Program {
    pub fn process_events(&mut self) -> Continuation {
        for event in self.event_loop.next(&mut self.glium_events_loop) {
            // Use the `winit` backend to convert the winit event to a conrod one.
            if let Some(ev) = conrod::backend::winit::convert_event(event.clone(), &self.display) {
                self.ui.handle_event(ev);
            };

            match event {
                glium::glutin::Event::WindowEvent { event, .. } => match event {
                    glium::glutin::WindowEvent::Closed => return Continuation::Stop,
                    _ => return Continuation::Continue,
                },
                _ => return Continuation::Continue,
            };
        }
        Continuation::Continue
    }

    pub fn new(title: &str, width: u32, height: u32, refresh_time: std::time::Duration) -> Program {
        let mut glium_events_loop = glium::glutin::EventsLoop::new();
        let window = glium::glutin::WindowBuilder::new()
            .with_title(title)
            .with_dimensions(width, height);
        let context = glium::glutin::ContextBuilder::new()
            .with_vsync(true)
            .with_multisampling(4);
        let display = glium::Display::new(window, context, &glium_events_loop).unwrap();
        Program {
            event_loop: EventLoop::new(refresh_time),
            ui: conrod::UiBuilder::new([width as f64, height as f64]).build(),
            glium_events_loop: glium_events_loop,
            renderer: conrod::backend::glium::Renderer::new(&display).unwrap(),
            display: display,
        }
    }

    pub fn draw(&mut self, f: &Fn(&mut conrod::UiCell) -> ()) -> () {
        // Process higher level events (DoubleClick ...) created by Ui::handle_event.
        let ui_cell = &mut self.ui.set_widgets();
        f(ui_cell)
    }

    pub fn render<Img>(&mut self, image_map: &conrod::image::Map<Img>) -> ()
    where
        Img: std::ops::Deref + conrod::backend::glium::TextureDimensions,
        for<'a> glium::uniforms::Sampler<'a, Img>: glium::uniforms::AsUniformValue,
    {
        if let Some(primitives) = self.ui.draw_if_changed() {
            self.renderer.fill(&self.display, primitives, image_map);
            let mut target = self.display.draw();
            target.clear_color(0.0, 0.0, 0.0, 1.0); // needs the Surface trait
            self.renderer
                .draw(&self.display, &mut target, image_map)
                .unwrap();
            target.finish().unwrap();
        }
    }
}

/// In most of the examples the `glutin` crate is used for providing the window context and
/// events while the `glium` crate is used for displaying `conrod::render::Primitives` to the
/// screen.
///
/// This `Iterator`-like type simplifies some of the boilerplate involved in setting up a
/// glutin+glium event loop that works efficiently with conrod.
pub struct EventLoop {
    time_step: std::time::Duration,
    last_update: std::time::Instant,
}

impl EventLoop {
    pub fn new(time_step: std::time::Duration) -> Self {
        EventLoop {
            time_step,
            last_update: std::time::Instant::now(),
        }
    }

    /// Produce an iterator yielding all available events.
    pub fn next(
        &mut self,
        events_loop: &mut glium::glutin::EventsLoop,
    ) -> Vec<glium::glutin::Event> {
        // We don't want to loop any faster than 60 FPS, so wait until it has been at least 16ms
        // since the last yield.
        let last_update = self.last_update;
        let duration_since_last_update = std::time::Instant::now().duration_since(last_update);
        if duration_since_last_update < self.time_step {
            std::thread::sleep(self.time_step - duration_since_last_update);
        }

        // Collect all pending events.
        let mut events = Vec::new();
        events_loop.poll_events(|event| events.push(event));

        // If there are no events and the `Ui` does not need updating, wait for the next event.
        if events.is_empty() {
            events_loop.run_forever(|event| {
                events.push(event);
                glium::glutin::ControlFlow::Break
            });
        }

        self.last_update = std::time::Instant::now();

        events
    }
}
