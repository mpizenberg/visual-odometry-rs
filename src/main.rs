#[macro_use]
extern crate serde_derive;

extern crate serde;
extern crate serde_json;

mod read_image;
mod json;

fn main() {
    json::main();
}
