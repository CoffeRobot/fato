#include "../include/profiler.h"
#include <sstream>

using namespace std;

namespace pinot_tracker {

Profiler::Profiler() {}

void Profiler::start(std::string id) {
  mutex_.lock();
  auto entry = profiler_.find(id);

  auto tp = chrono::high_resolution_clock::now();
  if (entry == profiler_.end()) {
    TimeEntry te;
    te.start_time = tp;
    profiler_.insert(pair<string, TimeEntry>(id, te));
  } else
    entry->second.start_time = tp;
  mutex_.unlock();
}

void Profiler::stop(std::string id) {
  mutex_.lock();
  auto entry = profiler_.find(id);

  if (entry == profiler_.end()) {
    mutex_.unlock();
    return;
  }

  auto tp = chrono::high_resolution_clock::now();
  auto begin = entry->second.start_time;

  entry->second.end_time = tp;
  entry->second.total_time +=
      chrono::duration_cast<chrono::milliseconds>(tp - begin).count();
  entry->second.num_calls++;
  mutex_.unlock();
}

float Profiler::getTime(string id) {
  mutex_.lock();
  auto entry = profiler_.find(id);

  if (entry == profiler_.end()) {
    mutex_.unlock();
    return 0;
  }

  float elapsed =
      entry->second.total_time / static_cast<float>(entry->second.num_calls++);
  mutex_.unlock();
  return elapsed;
}

string Profiler::getProfile() {
  mutex_.lock();
  stringstream ss;
  float overall = 0;
  for (auto it = profiler_.begin(); it != profiler_.end(); ++it) {
    float avg_time =
        it->second.total_time / static_cast<float>(it->second.num_calls);
    ss << it->first << ": " << avg_time << endl;
    overall += avg_time;
  }
  mutex_.unlock();
  ss << "-----> overall: " << overall << " <-----\n";
  return ss.str();
}

}  // end namespace
