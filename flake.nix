{
  description = "Optimised inference engine for Julia-1 on NVIDIA GPUs";

  inputs = {
    # Provides CUDA 13.2, LLVM 22, rust nightly, and cargo-oxide.
    cuda-oxide.url = "github:NVIDIA/cuda-rust/6d47ae49cb256a07948fb499c20c3eb9b2a96268?dir=cuda-oxide";
    nixpkgs.follows = "cuda-oxide/nixpkgs";
    flake-utils.follows = "cuda-oxide/flake-utils";
  };

  outputs =
    {
      cuda-oxide,
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          # CUDA, Rust, LLVM, cargo-oxide and the shellHook that finds the host driver.
          inputsFrom = [ cuda-oxide.devShells.${system}.default ];
        };
      }
    );
}
