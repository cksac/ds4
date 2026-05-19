cd ds4f
cargo build --release
cd ..
echo "c-ffi version"
./ds4f/target/release/ds4f run -p "hi" -m "/Users/sc/dev/ds4/ds4flash.gguf"

echo "rust-native version"
./ds4f/target/release/ds4f run -p "hi" -m "/Users/sc/dev/ds4/ds4flash.gguf" --native