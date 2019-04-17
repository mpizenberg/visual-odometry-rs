// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::{collections::HashMap, fs::File, io::Read};
use tar;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if let Err(error) = my_run(args[1].as_str()) {
        eprintln!("{:?}", error);
        std::process::exit(1);
    }
}

fn my_run(archive_path: &str) -> Result<(), std::io::Error> {
    let archive_file = File::open(archive_path)?;
    let mut archive = tar::Archive::new(archive_file);

    // Create a hashmap with the content
    let mut data = HashMap::new();
    for file in archive.entries()? {
        // Check for an I/O error.
        let mut file = file?;

        // Insert the file into the hashmap with its name as key.
        let file_path = file.header().path()?.to_str().expect("oops").to_owned();
        let mut buffer = Vec::with_capacity(file.header().size()? as usize);
        file.read_to_end(&mut buffer)?;
        data.insert(file_path, buffer);
    }

    for _ in 1..1000000000000_usize {
        println!("pause");
    }

    Ok(())
}
