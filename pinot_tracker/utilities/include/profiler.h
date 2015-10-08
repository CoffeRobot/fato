#ifndef PROFILER_H
#define PROFILER_H

#include <memory>
#include <map>
#include <chrono>
#include <string>
#include <mutex>

#ifdef TRACKER_WITH_GPU
#include <cuda_runtime.h>
#endif

namespace pinot_tracker{


#ifdef TRACKER_WITH_GPU

struct GpuTimeEntry
{
  cudaEvent_t start_time;
  cudaEvent_t end_time;
  float total_time;
  int num_calls;

  GpuTimeEntry():
    start_time(), end_time(), total_time(0.0f), num_calls(0)
  {};
};

#endif

struct TimeEntry
{
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    float total_time;
    int num_calls;

    TimeEntry():
        start_time(), end_time(), total_time(0.0f), num_calls(0)
    {};
};


class Profiler
{
public:
    static std::unique_ptr<Profiler>& getInstance()
    {
        // no need to check concurrency in c++11 when creating static object
        static std::unique_ptr<Profiler> instance(new Profiler());

        return instance;
    }

     ~Profiler();

    void start(std::string id);
    void stop(std::string id);
    void startGPU(std::string id);
    void stopGPU(std::string id);
    float getTime(std::string id);
    float getTimeGPU(std::string id);
    std::string getProfile();

private:
    Profiler();

    std::map<std::string,TimeEntry> profiler_;
    std::mutex mutex_;

    #ifdef TRACKER_WITH_GPU
    std::map<std::string,GpuTimeEntry> gpu_profiler_;
    #endif

};

}

#endif // PROFILER_H
