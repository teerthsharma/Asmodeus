import Lake
open Lake DSL

package «FCPA»

-- Path dependency on a local copy of Mathlib (v4.7.0 era); no network.
require mathlib from "./.lake/packages/mathlib"

@[default_target]
lean_lib «FCPA» where
  roots := #[`FCPA]
