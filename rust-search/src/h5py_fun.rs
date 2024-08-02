use std::fmt::Display;

use pyo3::{Python, types::PyDict, PyErr};
use numpy::{PyArray2};
use ndarray::Array2;

use crate::{Res, NoRes};
use crate::fs_fun::{create_path_to_file};

pub fn get_h5py_shape(file: &str, dataset: &str) -> Res<Vec<usize>> {
	Ok(Python::with_gil(|py| {
		let locals = PyDict::new(py);
		locals.set_item("h5py", py.import("h5py")?)?;
		py.eval(
			format!("h5py.File(\"{:}\")[\"{:}\"].shape", file, dataset).as_str(),
			None,
			Some(&locals)
		)?.extract()
	})?)
}
pub fn get_h5py_slice_f32(file: &str, dataset: &str, i_row_from: usize, i_row_to: usize) -> Res<Array2<f32>> {
	Ok(Python::with_gil(|py| {
		let locals = PyDict::new(py);
		locals.set_item("h5py", py.import("h5py")?)?;
		locals.set_item("np", py.import("numpy")?)?;
		locals.set_item("data", py.eval(
			format!("h5py.File(\"{:}\")[\"{:}\"]", file, dataset).as_str(),
			None,
			Some(&locals)
		)?)?;
		locals.set_item("start", i_row_from)?;
		locals.set_item("end", i_row_to)?;
		let slice_obj = py.eval(
			"np.array(data[start:end]).astype(np.float32)",
			None,
			Some(&locals)
		)?;
		let slice: &PyArray2<f32> = slice_obj.downcast()?;
		Ok::<_,PyErr>(slice.to_owned_array())
	})?)
}


pub struct H5PyBuilder<'py> {
	py: Python<'py>,
	locals: &'py PyDict,
}
impl<'py> H5PyBuilder<'py> {
	pub fn new(py: Python<'py>, file: &str) -> Res<H5PyBuilder<'py>> {
		let locals = PyDict::new(py);
		locals.set_item("h5py", py.import("h5py")?)?;
		locals.set_item("np", py.import("numpy")?)?;
		locals.set_item("file", py.eval(
			format!("h5py.File('{:}', 'w')", file).as_str(),
			None,
			Some(&locals)
		)?)?;
		Ok(H5PyBuilder{py: py, locals: locals})
	}
	pub fn with_attr_str(mut self, attr_name: &str, attr_value: &str) -> Res<Self> {
		self.with_attr(attr_name, format!("'{:}'", attr_value))
	}
	pub fn with_attr<F: Display>(mut self, attr_name: &str, attr_value: F) -> Res<Self> {
		self.py.eval(
			format!("file.attrs.create('{:}', {:})", attr_name, attr_value).as_str(),
			None,
			Some(&self.locals)
		)?;
		Ok(self)
	}
	pub fn with_dataset<T: numpy::Element>(mut self, dataset_name: &str, dataset: &Array2<T>, numpy_type_name: &str) -> Res<Self> {
		// self.locals.set_item("data", dataset.view().to_pyarray(self.py))?;
		unsafe {
			self.locals.set_item(
				"data",
				numpy::PyArray2::borrow_from_array(&dataset, self.locals)
			)?;
		}
		self.py.eval(
			format!("file.create_dataset('{:}', data.shape, dtype=data.dtype, data=data)", dataset_name).as_str(),
			None,
			Some(&self.locals)
		)?;
		Ok(self)
	}
	pub fn close(mut self) -> Res<Self> {
		self.py.eval("file.close()", None, Some(&self.locals))?;
		Ok(self)
	}
}

pub fn store_results(
	out_file: &str,
	kind: &str,
	size: &str,
	alg_name: &str,
	parameter_string: &str,
	neighbor_dists: &Array2<f32>,
	neighbor_ids: &Array2<usize>,
	data_bin: &Array2<u64>,
	query_bin: &Array2<u64>,
	build_time: f64,
	query_time: f64,
) -> NoRes {
	create_path_to_file(out_file)?;
	Python::with_gil(|py| {
		H5PyBuilder::new(py, out_file)?
		.with_dataset("dists", neighbor_dists, "float32")?
		.with_dataset("knns", neighbor_ids, "uint64")?
		.with_dataset("db", data_bin, "uint64")?
		.with_dataset("queries", query_bin, "uint64")?
		.with_attr_str("algo", alg_name)?
		.with_attr_str("data", kind)?
		.with_attr_str("size", size)?
		.with_attr_str("params", parameter_string)?
		.with_attr("buildtime", build_time)?
		.with_attr("querytime", query_time)?
		.close()?;
		Ok::<(),Box<dyn std::error::Error>>(())
	})?;
	Ok(())
}

pub fn store_results_no_sketches(
	out_file: &str,
	kind: &str,
	size: &str,
	alg_name: &str,
	parameter_string: &str,
	neighbor_dists: &Array2<f32>,
	neighbor_ids: &Array2<usize>,
	data_bin: &Array2<u64>,
	query_bin: &Array2<u64>,
	build_time: f64,
	query_time: f64,
) -> NoRes {
	create_path_to_file(out_file)?;
	Python::with_gil(|py| {
		H5PyBuilder::new(py, out_file)?
		.with_dataset("dists", neighbor_dists, "float32")?
		.with_dataset("knns", neighbor_ids, "uint64")?
		// .with_dataset("db", data_bin, "uint64")?
		// .with_dataset("queries", query_bin, "uint64")?
		.with_attr_str("algo", alg_name)?
		.with_attr_str("data", kind)?
		.with_attr_str("size", size)?
		.with_attr_str("params", parameter_string)?
		.with_attr("buildtime", build_time)?
		.with_attr("querytime", query_time)?
		.close()?;
		Ok::<(),Box<dyn std::error::Error>>(())
	})?;
	Ok(())
}

pub fn store_results_hamming(
	out_file: &str,
	kind: &str,
	size: &str,
	alg_name: &str,
	parameter_string: &str,
	neighbor_dists: &Array2<usize>,
	neighbor_ids: &Array2<usize>,
	data_bin: &Array2<u64>,
	query_bin: &Array2<u64>,
	build_time: f64,
	query_time: f64,
) -> NoRes {
	create_path_to_file(out_file)?;
	Python::with_gil(|py| {
		H5PyBuilder::new(py, out_file)?
		.with_dataset("dists", neighbor_dists, "uint64")?
		.with_dataset("knns", neighbor_ids, "uint64")?
		.with_dataset("db", data_bin, "uint64")?
		.with_dataset("queries", query_bin, "uint64")?
		.with_attr_str("algo", alg_name)?
		.with_attr_str("data", kind)?
		.with_attr_str("size", size)?
		.with_attr_str("params", parameter_string)?
		.with_attr("buildtime", build_time)?
		.with_attr("querytime", query_time)?
		.close()?;
		Ok::<(),Box<dyn std::error::Error>>(())
	})?;
	Ok(())
}

pub fn store_results_hamming_no_sketches(
	out_file: &str,
	kind: &str,
	size: &str,
	alg_name: &str,
	parameter_string: &str,
	neighbor_dists: &Array2<usize>,
	neighbor_ids: &Array2<usize>,
	data_bin: &Array2<u64>,
	query_bin: &Array2<u64>,
	build_time: f64,
	query_time: f64,
) -> NoRes {
	create_path_to_file(out_file)?;
	Python::with_gil(|py| {
		H5PyBuilder::new(py, out_file)?
		.with_dataset("dists", neighbor_dists, "uint64")?
		.with_dataset("knns", neighbor_ids, "uint64")?
		// .with_dataset("db", data_bin, "uint64")?
		// .with_dataset("queries", query_bin, "uint64")?
		.with_attr_str("algo", alg_name)?
		.with_attr_str("data", kind)?
		.with_attr_str("size", size)?
		.with_attr_str("params", parameter_string)?
		.with_attr("buildtime", build_time)?
		.with_attr("querytime", query_time)?
		.close()?;
		Ok::<(),Box<dyn std::error::Error>>(())
	})?;
	Ok(())
}


