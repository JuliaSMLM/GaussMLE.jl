# Testing Guidelines for test/

GaussMLE follows the lab's standard test layout (admiral decisions 0007 to 0010).

## Layout

| Group | Folder | Contents | Runs on |
|---|---|---|---|
| Core | `test/*.jl` | API smoke, CPU kernel, sCMOS variance-map indexing regression | GitHub CI (Julia min and 1); at most 2 min |
| QA | `test/qa/` | Aqua and ExplicitImports | GitHub CI, its own job |
| GPU | `test/gpu/` | GPU kernel, CPU vs GPU agreement, performance benchmark with std/CRLB checks | lab GPU machine |
| Long | `test/long/` | Monte Carlo validation: bias and std vs CRLB for every model and camera | lab machine |

- `test/runtests.jl` is the lab template, identical in every package: never edit it.
- `test/test_groups.toml` declares the groups and the heavy work (commands, resources, time).
- `test/qa/qa.jl` is the lab template; a check is opted out only with the reason beside it.
- Each file runs in its own module and `@testset`, so each file starts with its own `using` lines.
- Helpers a file `include`s go in a `utils/` subfolder of its group folder (not run as tests).
  `test/long/utils/validation_utils.jl` is also included by the GPU benchmark.

## Running

```bash
julia --project -e 'using Pkg; Pkg.test()'                   # Core
GROUP=QA julia --project -e 'using Pkg; Pkg.test()'          # one group; "GPU,Long" for several
GROUP=Everything julia --project -e 'using Pkg; Pkg.test()'  # every group this machine can run
```

## Rules

- **No @test_skip**: a failing test is information; fix the code or the test.
- Core stays under 2 minutes on a GitHub runner, counting first-call compile. A test that pushes
  it over moves to Long (CPU) or GPU.
- New or changed public behaviour gets tests of its documented contract; a bug fix gets one
  regression test. No tests of private helpers, no duplicates of behaviour tested elsewhere.
- Report the test count per group before and after every change.
- Statistical tests seed their RNG or compare on a tolerance that holds for any draw.
