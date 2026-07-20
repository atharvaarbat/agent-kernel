import Lake
open Lake DSL

package «rvt» where
  name := "rvt"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.14.0"

lean_lib «RVT» where
  roots := #[`RVT]
