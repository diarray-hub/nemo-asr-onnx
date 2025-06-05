/// Utility functions for tensor shapes.

/// Returns a shape list for a [1, channels, frames] tensor.
List<int> shape1xCxT(int channels, int frames) => [1, channels, frames];
