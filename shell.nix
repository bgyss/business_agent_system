# Legacy shell.nix for backwards compatibility
# Use `nix develop` or `nix-shell` to enter the development environment
(import (
  let
    lock = builtins.fromJSON (builtins.readFile ./flake.lock);
    nodeName = lock.nodes.root.inputs.nixpkgs;
  in fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/${lock.nodes.${nodeName}.locked.rev}.tar.gz";
    sha256 = lock.nodes.${nodeName}.locked.narHash;
  }
) {}).callPackage ({ mkShell, ... }: mkShell {
  inputsFrom = [ (import ./flake.nix).outputs.devShells.x86_64-linux.default ];
}) {}