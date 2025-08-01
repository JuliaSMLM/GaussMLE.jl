# Testing Guidelines for test/

This directory contains all tests for the package. Follow these conventions when writing or modifying tests.

## Test Structure

### runtests.jl Organization
- **Only contains**: `using` statements and the overall test structure
- **No test logic**: All actual tests are included from other files
- **All imports here**: Any packages needed for testing must be imported at the top of runtests.jl

### Test File Organization
1. **User-facing API tests** (e.g., `test_api.jl`)
   - Tests all exported functions that users interact with
   - Tests various keyword arguments and options
   - Focuses on expected use cases and behavior

2. **Internal function tests** (organized by module/concept)
   - Separate files for different modules or logical groupings
   - Tests internal functions and implementation details
   - Clear naming scheme (e.g., `test_utils.jl`, `test_parser.jl`)

### Important Rules
- **No using statements in included files** - All imports must be in runtests.jl
- **Aim for simplicity** - Good coverage without bloating tests
- **Avoid pedantic edge cases** - Focus on meaningful tests that aid development
- **Maintainability first** - Tests should be easy to update as code evolves

## Running Tests

### From Julia REPL
```julia
# Activate the project (from package root)
using Pkg
Pkg.activate(".")

# Run all tests
Pkg.test()

# Or with package name
Pkg.test("GaussMLE")
```

### During Development
```julia
# Run specific test file directly
include("test/runtests.jl")

# Or run a specific test file
include("test/test_specific.jl")  # Only works if no using statements needed
```

## Writing New Tests
- Group related tests in `@testset` blocks with descriptive names
- Use meaningful test descriptions
- Test both success cases and expected failures
- Keep tests focused and independent

## GPU Testing
To run GPU tests, set the environment variable:
```bash
GAUSSMLE_TEST_GPU=true julia --project test/gpu_tests.jl
```