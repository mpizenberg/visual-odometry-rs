use conrod::backend::glium::glium;
use std;

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
