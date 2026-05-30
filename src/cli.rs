//! Command-line argument parsing for the inference binary.
//!
//! Accepts one or two positional safetensors paths:
//! ```text
//! phi3-mixed-quant <packed-model.safetensors> [shard2.safetensors]
//! ```

use std::path::PathBuf;

const USAGE: &str = "Usage: phi3-mixed-quant <packed-model.safetensors> [shard2.safetensors]\n\
     Single file: phi3-mixed-quant model.safetensors\n\
     Two shards : phi3-mixed-quant shard1.safetensors shard2.safetensors";

#[derive(Debug, Clone)]
pub struct CliArgs {
    pub shard1: PathBuf,
    pub shard2: Option<PathBuf>,
}

impl CliArgs {
    /// Parse `std::env::args()`, skipping the binary name.
    pub fn parse() -> candle::Result<Self> {
        Self::parse_from(std::env::args().skip(1))
    }

    /// Parse from any iterator of arguments (handy for tests).
    pub fn parse_from<I, S>(args: I) -> candle::Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut it = args.into_iter().map(Into::into);
        let shard1: PathBuf = it
            .next()
            .ok_or_else(|| candle::Error::Msg(USAGE.to_string()))?
            .into();
        let shard2 = it.next().map(PathBuf::from);
        Ok(Self { shard1, shard2 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_shard() {
        let a = CliArgs::parse_from(["model.safetensors"]).unwrap();
        assert_eq!(a.shard1, PathBuf::from("model.safetensors"));
        assert!(a.shard2.is_none());
    }

    #[test]
    fn parses_two_shards() {
        let a = CliArgs::parse_from(["a.st", "b.st"]).unwrap();
        assert_eq!(a.shard2, Some(PathBuf::from("b.st")));
    }

    #[test]
    fn rejects_empty() {
        assert!(CliArgs::parse_from(std::iter::empty::<String>()).is_err());
    }
}
