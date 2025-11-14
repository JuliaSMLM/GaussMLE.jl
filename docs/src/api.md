# API Reference

```@index
```

## Public API

### Main Types and Functions

```@docs
GaussMLEFitter
fit
GaussMLEResults
```

### PSF Models

```@docs
PSFModel
GaussianXYNB
GaussianXYNBS
GaussianXYNBSXSY
AstigmaticXYZNB
```

### Camera Models

```@docs
CameraModel
IdealCamera
SCMOSCameraInternal
to_electrons
get_variance_map
```

### Data Structures

```@docs
ROIBatch
SingleROI
LocalizationResult
```

### Constraints

```@docs
ParameterConstraints
default_constraints
```

### Simulation

```@docs
generate_roi_batch
```

### Device Management

```@docs
ComputeDevice
CPU
GPU
auto_device
select_device
```

## Internal API

```@autodocs
Modules = [GaussMLE]
Public = false
```