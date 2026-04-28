//! Inference binary entry point.
//!
//! Argument parsing lives in [`phi3_mixed_quant::cli`], the chat loop in
//! [`phi3_mixed_quant::repl`], and the token-by-token generation in
//! [`phi3_mixed_quant::generation`]. This file just wires them together.

use phi3_mixed_quant::{cli::CliArgs, repl};

fn main() -> candle::Result<()> {
    let args = CliArgs::parse()?;
    repl::run(args)
}
