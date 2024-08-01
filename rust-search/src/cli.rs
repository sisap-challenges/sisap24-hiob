use clap::Parser;

#[derive(Parser, Debug)]
pub struct Cli {
	/// Input base path
	#[arg(short, long, default_value="../data")]
	pub in_path: String,
	/// Output base path
	#[arg(short, long, default_value="../result")]
	pub out_path: String,
	/// The size tag (e.g. "100K") to run experiments for
	#[arg(long, default_value="100K")]
	pub size: String,
	/// The number of nearest neighbors to search
	#[arg(long, default_value_t=10)]
	pub k: usize,
	/// Whether or not to load the entire dataset into RAM. If false, will only load as much data from disk at any time as required for execution
	#[arg(long, default_value_t=false)]
	pub ram: bool,
	/// How many iterations to run the Stochastic HIOB
	#[arg(long, default_value_t=30000)]
	pub its: usize,
	/// How many bits to use in binarization
	#[arg(long, default_value="1024")]
	pub bits: String,
	/// How many iterations to run per stochastic sub sample
	#[arg(long, default_value_t=1024)]
	pub batch_its: usize,
	/// How many samples to use in each stochastic batch
	#[arg(long, default_value_t=10000)]
	pub samples: usize,
	/// Standard deviation of noise to add to stochastic subsamples
	#[arg(long, default_value_t=0.0)]
	pub noise: f32,
	/// Minimum number of probes to evaluate in queries
	#[arg(long, default_value_t=10)]
	pub probe_min: usize,
	/// Maximum number of probes to evaluate in queries
	#[arg(long, default_value_t=100)]
	pub probe_max: usize,
	/// Number of probe values (logarithmically interpolated) to evaluate in queries including min and max probes
	#[arg(long, default_value_t=11)]
	pub probe_steps: usize,
	/// Whether or not to run with refining, i.e. computing effective distances on true vectors
	#[arg(long, default_value_t=false)]
	pub refine: bool,
	/// Number of virtual cores *not* to use by this program
	#[arg(long, default_value_t=0)]
	pub idle_cpus: usize,
	/// Whether to use ITQ initialization, if false will use RANSAC
	#[arg(long, default_value_t=false)]
	pub itq: bool,
}

