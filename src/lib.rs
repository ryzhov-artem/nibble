//! Library facade for `phi3-mixed-quant`.
//!
//! Both the inference binary (`main.rs`) and the offline tools
//! (`bin/quantize_q8k.rs`, `bin/pack_q8k_safetensors.rs`) depend on the
//! shared on-disk header definition (`types::Q8KHeader`) and magic numbers.
//! Exposing the modules here lets every binary `use phi3_mixed_quant::...`
//! instead of triplicating the layout structs.

pub mod cache;
pub mod cli;
pub mod conversation;
pub mod generation;
pub mod loader;
pub mod model;
pub mod quant_linear;
pub mod repl;
pub mod scratch;
pub mod types;
