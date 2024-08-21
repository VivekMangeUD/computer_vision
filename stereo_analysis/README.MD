# Stereo Analysis System Project

**Author:** Vivek Mange

This project involves designing and implementing a stereo analysis system using region-based and feature-based matching methods. The system performs multi-resolution stereo analysis, allowing for customizable levels, template sizes, and search neighborhoods.

### Concepts:

1. **Region-Based Analysis:** User-defined template and matching window sizes are used to compare image regions with metrics like SAD, SSD, and NCC.

2. **Feature-Based Analysis:** Harris corner detection extracts features, with matching performed using SAD, SSD, or NCC.

3. **Multi-Resolution Matching:** Stereo matching is performed at multiple levels, with disparity from each level initializing the next.

4. **Validity Check:** Ensures consistency between left-to-right and right-to-left matching, filling gaps with interpolated values if discrepancies arise.

5. **Data:** Uses Middlebury Stereo datasets for evaluation.

### Output:

- **Matching Results:** Provide at least 3 results for 5 stereo pairs.
- **Customization:** Users can set parameters for template size, search neighborhood, and matching method.

