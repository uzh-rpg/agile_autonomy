#include <fstream>

// print trajectory
void print_trajectory(const float* trajectory, const int traj_len) {
  for (int j = 0; j < traj_len; j++) {
    printf(
        "P: %.3f, %.3f, %.3f | V: %.3f, %.3f, %.3f | Q: %.3f, %.3f, %.3f, %.3f "
        "| U: %.3f, %.3f, %.3f, %.3f\n",
        trajectory[14 * j + 0], trajectory[14 * j + 1], trajectory[14 * j + 2],
        trajectory[14 * j + 3], trajectory[14 * j + 4], trajectory[14 * j + 5],
        trajectory[14 * j + 6], trajectory[14 * j + 7], trajectory[14 * j + 8],
        trajectory[14 * j + 9], trajectory[14 * j + 10],
        trajectory[14 * j + 11], trajectory[14 * j + 12],
        trajectory[14 * j + 13]);
  }
}