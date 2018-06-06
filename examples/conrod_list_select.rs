//! Select list using conrod

#[macro_use]
extern crate conrod;

mod program;

use conrod::backend::glium::glium;
use conrod::{widget, Borderable, Colorable, Labelable, Positionable, Sizeable, Widget};

const WIDTH: u32 = 620;
const HEIGHT: u32 = 480;
const FONT_PATH: &str = "data/fonts/NotoSans/NotoSans-Regular.ttf";

fn main() {
    // Init the program
    let mut prog = program::Program::new(
        "Conrod select list",
        WIDTH,
        HEIGHT,
        std::time::Duration::from_millis(16),
    );

    // Add a `Font` to the `Ui`'s `font::Map` from file.
    prog.ui.fonts.insert_from_file(FONT_PATH).unwrap();

    // The hash map containing our images (none here).
    let image_map = conrod::image::Map::<glium::texture::Texture2d>::new();

    // List of entries to display. They should implement the Display trait.
    let list_items = [
        "African Sideneck Turtle".to_string(),
        "Alligator Snapping Turtle".to_string(),
        "Common Snapping Turtle".to_string(),
        "Indian Peacock Softshelled Turtle".to_string(),
        "Eastern River Cooter".to_string(),
        "Eastern Snake Necked Turtle".to_string(),
        "Diamond Terrapin".to_string(),
        "Indian Peacock Softshelled Turtle".to_string(),
        "Musk Turtle".to_string(),
        "Reeves Turtle".to_string(),
        "Eastern Spiny Softshell Turtle".to_string(),
        "Red Ear Slider Turtle".to_string(),
        "Indian Tent Turtle".to_string(),
        "Mud Turtle".to_string(),
        "Painted Turtle".to_string(),
        "Spotted Turtle".to_string(),
    ];

    // List of selections, should be same length as list of entries.
    // Will be updated by the widget.
    let mut list_selected = std::collections::HashSet::new();

    // Create our widgets.
    widget_ids!(struct Ids { canvas, list_select });
    let ids = Ids::new(prog.ui.widget_id_generator());

    let mut my_widgets = |ui: &mut conrod::UiCell| {
        // Create a background canvas upon which we'll place the button.
        widget::Canvas::new()
            .color(conrod::color::BLUE)
            .set(ids.canvas, ui);

        // Instantiate the `ListSelect` widget.
        let num_items = list_items.len();
        let item_h = 30.0;
        let font_size = item_h as conrod::FontSize / 2;
        let (mut events, scrollbar) = widget::ListSelect::multiple(num_items)
            .flow_down()
            .item_size(item_h)
            .scrollbar_next_to()
            .w_h(400.0, 230.0)
            .top_left_with_margins_on(ids.canvas, 40.0, 40.0)
            .set(ids.list_select, ui);

        // Handle the `ListSelect`s events.
        while let Some(event) = events.next(ui, |i| list_selected.contains(&i)) {
            use conrod::widget::list_select::Event;
            match event {
                // For the `Item` events we instantiate the `List`'s items.
                Event::Item(item) => {
                    let label = &list_items[item.i];
                    let (color, label_color) = match list_selected.contains(&item.i) {
                        true => (conrod::color::LIGHT_BLUE, conrod::color::YELLOW),
                        false => (conrod::color::LIGHT_GREY, conrod::color::BLACK),
                    };
                    let button = widget::Button::new()
                        .border(0.0)
                        .color(color)
                        .label(label)
                        .label_font_size(font_size)
                        .label_color(label_color);
                    item.set(button, ui);
                }

                // The selection has changed.
                Event::Selection(selection) => {
                    selection.update_index_set(&mut list_selected);
                    println!("selected indices: {:?}", list_selected);
                }

                // The remaining events indicate interactions with the `ListSelect` widget.
                event => println!("{:?}", &event),
            }
        }

        // Instantiate the scrollbar for the list.
        if let Some(s) = scrollbar {
            s.set(ui);
        }
    };

    // Run forever our program.
    prog.run(&image_map, &mut my_widgets);
}
