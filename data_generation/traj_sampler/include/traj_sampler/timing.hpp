#pragma once

#include <assert.h>
#include <math.h>
#include <chrono>
#include <limits>
#include <mutex>
#include <string>

namespace traj_sampler {

template <typename T = double>
class Timing {
 public:
  Timing(const bool thread_safe = false) : thread_safe_(thread_safe) {}

  Timing(const Timing<T>& other) : thread_safe_(other.thread_safe_) {
    if (thread_safe_) other.mutex_.lock();
    ticked_ = other.ticked_;
    t_start_ = other.t_start_;
    timing_mean_ = other.timing_mean_;
    timing_last_ = other.timing_last_;
    timing_S_ = other.timing_S_;
    timing_min_ = other.timing_min_;
    timing_max_ = other.timing_max_;
    n_samples_ = other.n_samples_;
    if (thread_safe_) other.mutex_.unlock();
  }

  ~Timing() {}
  Timing<T>& operator=(const Timing<T>& other) {
    assert(thread_safe_ == other.thread_safe_);

    if (&other != this) {
      std::unique_lock<std::mutex> lock_this(mutex_, std::defer_lock);
      std::unique_lock<std::mutex> lock_other(other.mutex_, std::defer_lock);
      if (other.thread_safe_) std::lock(lock_this, lock_other);

      ticked_ = other.ticked_;
      t_start_ = other.t_start_;
      timing_mean_ = other.timing_mean_;
      timing_last_ = other.timing_last_;
      timing_S_ = other.timing_S_;
      timing_min_ = other.timing_min_;
      timing_max_ = other.timing_max_;
      n_samples_ = other.n_samples_;
    }

    return *this;
  }

  // tic set the start time.
  void tic() {
    if (thread_safe_) mutex_.lock();
    t_start_ = std::chrono::high_resolution_clock::now();
    ticked_ = true;
    if (thread_safe_) mutex_.unlock();
  }

  // toc does timing, fitlers and saves values, and resets start time.
  // If the argument multi_toc is set true,
  // toc can be called multiple times in sequence without calling tic.
  T toc(const bool multi_toc = false) {
    if (!ticked_ && !multi_toc) return std::numeric_limits<T>::min();

    if (thread_safe_) mutex_.lock();

    // Calculate timing.
    const TimePoint t_end = std::chrono::high_resolution_clock::now();
    timing_last_ = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(
                              t_end - t_start_)
                              .count();
    n_samples_++;

    // Set timing, filter if already initialized.
    if (timing_mean_ <= 0.0) {
      timing_mean_ = timing_last_;
    } else {
      T timing_mean_prev = timing_mean_;
      timing_mean_ =
          timing_mean_prev + (timing_last_ - timing_mean_prev) / n_samples_;
      timing_S_ = timing_S_ + (timing_last_ - timing_mean_prev) *
                                  (timing_last_ - timing_mean_);
    }
    timing_min_ = (timing_last_ < timing_min_) ? timing_last_ : timing_min_;
    timing_max_ = (timing_last_ > timing_max_) ? timing_last_ : timing_max_;
    timing_total_ += timing_last_;

    // Reset start time, to allow multiple tocs, useful to not increase counter.
    t_start_ = t_end;

    if (thread_safe_) mutex_.unlock();

    return timing_mean_;
  }

  // Reset saved timings and calls;
  void reset() {
    if (thread_safe_) mutex_.lock();
    n_samples_ = 0u;
    t_start_ = TimePoint();
    timing_mean_ = 0.0;
    timing_last_ = 0.0;
    timing_S_ = 0.0;
    timing_min_ = std::numeric_limits<T>::max();
    timing_max_ = 0.0;
    timing_total_ = 0.0;
    if (thread_safe_) mutex_.unlock();
  }

  // Accessors
  T timing() const { return timing_mean_; }
  T timing_mean() const { return timing_mean_; }
  T timing_last() const { return timing_last_; }
  T timing_min() const { return timing_min_; }
  T timing_max() const { return timing_max_; }
  T timing_std() const { return std::sqrt(timing_S_ / n_samples_); }
  T timing_total() const { return timing_total_; }
  uint64_t count() const { return n_samples_; }

  // Print timing information to console with optional name tag.
  void print(const std::string& name) const {
    T timing_mean, timing_S, timing_min, timing_max;
    uint64_t n_samples;

    if (thread_safe_) mutex_.lock();
    timing_mean = timing_mean_;
    timing_S = timing_S_;
    timing_min = timing_min_;
    timing_max = timing_max_;
    n_samples = n_samples_;
    if (thread_safe_) mutex_.unlock();

    if (n_samples_ < 1) {
      std::printf("Timing %s\n   no call yet.\n", name.c_str());
    } else {
      std::printf(
          "Timing %s in %lu calls\n"
          "mean|std:%9.3f |%9.3f ms  [min|max:%9.3f |%9.3f ms]\n",
          name.c_str(), n_samples, 1000 * timing_mean, 1000 * timing_S,
          1000 * timing_min, 1000 * timing_max);
    }
  }

 private:
  using TimePoint = std::chrono::high_resolution_clock::time_point;
  bool ticked_{false};
  TimePoint t_start_;

  // Initialize timing to impossible values.
  T timing_mean_{0.0};
  T timing_last_{0.0};
  T timing_S_{0.0};
  T timing_min_{std::numeric_limits<T>::max()};
  T timing_max_{0.0};
  T timing_total_{0.0};

  uint64_t n_samples_{0u};

  // Thread safety.
  const bool thread_safe_;
  mutable std::mutex mutex_;
};

}  // namespace mppi