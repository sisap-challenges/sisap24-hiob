use std::fs::{create_dir_all, File};
use std::path::Path;
use std::io::copy;

use crate::NoRes;

pub fn create_path_to_file(file_path: &str) -> NoRes {
	let path = Path::new(file_path);
	let prefix = path.parent().unwrap();
	create_dir_all(prefix)?;
	Ok(())
}


// Download a single h5 file to the specified destination if not already available
pub fn download_if_missing(file_url: &str, file_path: &str) -> NoRes {
	if Path::new(file_path).exists() {
		println!("file '{}' already at '{}'", file_url, file_path);
		Ok(())
	} else {
		/* Create parent directories */
		create_path_to_file(file_path)?;
		/* Start download */
		println!("downloading '{}' -> '{}'...", file_url, file_path);
		let response = reqwest::blocking::get(file_url)?;
		let mut dest = File::create(file_path)?;
		let mut content =  std::io::Cursor::new(response.bytes()?);
		copy(&mut content, &mut dest)?;
		Ok(())
	}
}