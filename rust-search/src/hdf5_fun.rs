use hdf5::{File, Group, H5Type, types::{VarLenUnicode}};
use std::ops::Deref;
use ndarray::{Array2};

use crate::{Res, NoRes};
use crate::fs_fun::{create_path_to_file};

pub fn create_str_attr<T: Deref<Target=Group>>(location: &T, name: &str, value: &str) -> NoRes {
	let attr = location.new_attr::<VarLenUnicode>().create(name)?;
	let value_: VarLenUnicode = value.parse()?;
	attr.write_scalar(&value_)?;
	Ok(())
}
pub fn create_num_attr<T: Deref<Target=Group>, F: H5Type>(location: &T, name: &str, value: F) -> NoRes {
	let attr = location.new_attr::<VarLenUnicode>().create(name)?;
	attr.write_scalar(&value)?;
	Ok(())
}

pub struct H5Builder {
	file: File
}
impl H5Builder {
	pub fn new(file: &str) -> Res<H5Builder> {
		create_path_to_file(file)?;
		Ok(H5Builder{
			file: File::create(file)?
		})
	}
	pub fn with_str_attr(mut self, attr_name: &str, attr_value: &str) -> Res<Self> {
		create_str_attr(&self.file, attr_name, attr_value)?;
		Ok(self)
	}
	pub fn with_num_attr<F: H5Type>(mut self, attr_name: &str, attr_value: F) -> Res<Self> {
		create_num_attr(&self.file, attr_name, attr_value)?;
		Ok(self)
	}
	pub fn with_dataset<F: H5Type>(mut self, dataset_name: &str, dataset: &Array2<F>) -> Res<Self> {
		self.file.new_dataset_builder().with_data(dataset.view()).create(dataset_name)?;
		Ok(self)
	}
}

pub fn open_hdf5(file_path: &str) -> Res<File> {
	let result = File::open(file_path)?;
	Ok(result)
}


// Store the output values as required in the task specification
fn store_results<T: H5Type>(
	out_file: &str,
	kind: &str,
	size: &str,
	alg_name: &str,
	parameter_string: &str,
	neighbor_dists: Array2<T>,
	neighbor_ids: Array2<usize>,
	data_bin: &Array2<u64>,
	query_bin: &Array2<u64>,
	build_time: f64,
	query_time: f64,
) -> NoRes {
	H5Builder::new(out_file)?
	.with_dataset("dists", &neighbor_dists)?
	.with_dataset("knns", &neighbor_ids)?
	.with_dataset("db", neighbor_dists)?
	.with_dataset("queries", neighbor_ids)?
	.with_str_attr("algo", alg_name)?
	.with_str_attr("data", kind)?
	.with_str_attr("size", size)?
	.with_str_attr("params", parameter_string)?
	.with_num_attr("buildtime", build_time)?
	.with_num_attr("querytime", query_time)?;
	Ok(())
}