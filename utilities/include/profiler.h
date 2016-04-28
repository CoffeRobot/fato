/*****************************************************************************/
/*  Copyright (c) 2016, Alessandro Pieropan                                  */
/*  All rights reserved.                                                     */
/*                                                                           */
/*  Redistribution and use in source and binary forms, with or without       */
/*  modification, are permitted provided that the following conditions       */
/*  are met:                                                                 */
/*                                                                           */
/*  1. Redistributions of source code must retain the above copyright        */
/*  notice, this list of conditions and the following disclaimer.            */
/*                                                                           */
/*  2. Redistributions in binary form must reproduce the above copyright     */
/*  notice, this list of conditions and the following disclaimer in the      */
/*  documentation and/or other materials provided with the distribution.     */
/*                                                                           */
/*  3. Neither the name of the copyright holder nor the names of its         */
/*  contributors may be used to endorse or promote products derived from     */
/*  this software without specific prior written permission.                 */
/*                                                                           */
/*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS      */
/*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT        */
/*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR    */
/*  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT     */
/*  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,   */
/*  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT         */
/*  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    */
/*  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY    */
/*  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT      */
/*  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE    */
/*  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.     */
/*****************************************************************************/

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

namespace fato{


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
