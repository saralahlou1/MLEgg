{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
  flake-utils.lib.eachDefaultSystem (system:
    let pkgs = nixpkgs.legacyPackages.${system}; in
    {
      devShells = {
        egg = pkgs.mkShell {
          name = "egg";

          packages = with pkgs; [ cargo rustc rust-analyzer rustfmt ];

          shellHook = ''
            exec ${pkgs.zsh}/bin/zsh
          '';
        };

        mlir = pkgs.mkShell {
          name = "mlir";

          packages = with pkgs;
            [ meson pkg-config cmake ninja libxml2 clang-tools cmake-format ] ++ (with llvmPackages; [
              llvm
              (mlir.overrideAttrs (old: { outputs = [ "out" ]; })) # if we don't do this then cmake just doesn't find things
            ]);

          shellHook = ''
            exec ${pkgs.zsh}/bin/zsh
          '';
        };

      };
    });
}
