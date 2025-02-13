cargo +nightly fmt -- --check --color always
cargo +nightly clippy --all-targets -- -D warnings
