# GaussMLE coord-convention-fix Branch - Progress Summary

## Completed Today (6 commits)

1. **Fix sCMOS bounds errors and infinite recursion**
   - Fixed 6 tests with out-of-bounds corners
   - Added bounds validation to simulator
   - Fixed SCMOSCamera recursion bug

2. **Refactor to Option 3: Remove fitter.camera_model**
   - Eliminated duplicate IdealCamera types
   - Simplified API: no camera_model parameter  
   - Clean dispatch on ROIBatch type only

3. **Fix Val() dispatch**
   - Proper compile-time type creation

4. **Fix coordinate extraction with ROIBatch corners**
   - extract_roi_coords now requires actual corners
   - validate_fits takes ROIBatch parameter
   - Fixed 561334x coordinate mismatch → 0.0006 pixels

5. **Fix sCMOS variance indexing**
   - Kernel now takes corners parameter
   - Converts ROI indices to camera coords
   - Variance map stays 2D (camera property)

## Test Status

**Passing:**
- sCMOS Camera with Variance Map: 3/3 ✓
- Edge Cases: 4/4 ✓
- Basic functionality works

**Failing:**
- Many std_pass checks (std/CRLB ratio validation)
- Astigmatic model: 10 failures
- GPU tests: 5 errors (not failures - actual errors)

## Root Issues Remaining

1. **GPU kernel errors** - need error messages to diagnose
2. **Statistical validation failures** - std/CRLB ratios off
3. **Possible coordinate issues still present** - many models failing

## Recommendation

This is a good checkpoint. The branch has significant improvements but needs systematic debugging of remaining issues. Suggest:

1. Focus on getting ONE test file to 100% pass
2. Fix GPU errors (likely kernel signature issue)
3. Debug std/CRLB calculation issues

Total changes: 9 files, 380 insertions, 190 deletions
