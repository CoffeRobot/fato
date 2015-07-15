#ifndef PROFILER_H
#define PROFILER_H

#include <memory>
#include <map>
#include <chrono>
#include <string>
#include <mutex>

namespace pinot_tracker{

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

    void start(std::string id);
    void stop(std::string id);
    float getTime(std::string id);
    std::string getProfile();

private:
    Profiler();



    std::map<std::string,TimeEntry> profiler_;
    std::mutex mutex_;
};

}

#endif // PROFILER_H
