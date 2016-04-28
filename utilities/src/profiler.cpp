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

#include "../include/profiler.h"
#include <sstream>

using namespace std;

namespace fato {

Profiler::Profiler() {}

Profiler::~Profiler()
{
#ifdef TRACKER_WITH_GPU
    for(auto it = gpu_profiler_.begin(); it != gpu_profiler_.end(); it++)
    {
        cudaEventDestroy(it->second.start_time);
        cudaEventDestroy(it->second.end_time);
    }
#endif
}

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

void Profiler::startGPU(string id)
{
#ifdef TRACKER_WITH_GPU
  mutex_.lock();
  auto entry = gpu_profiler_.find(id);

  if (entry == gpu_profiler_.end()) {
    GpuTimeEntry te;
    cudaEventCreate(&te.start_time);
    cudaEventCreate(&te.end_time);
    cudaEventRecord(te.start_time, 0);
    gpu_profiler_.insert(pair<string, GpuTimeEntry>(id, te));
  } else {
    cudaEventRecord(entry->second.start_time, 0);
  }
  mutex_.unlock();
#endif
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

void Profiler::stopGPU(string id)
{
#ifdef TRACKER_WITH_GPU
  mutex_.lock();
  auto entry = gpu_profiler_.find(id);

  if (entry == gpu_profiler_.end()) {
    mutex_.unlock();
    return;
  }

  cudaEventRecord(entry->second.end_time, 0);
  auto begin = entry->second.start_time;

  float elapsed;
  cudaEventSynchronize(entry->second.end_time);
  cudaEventElapsedTime(&elapsed, entry->second.start_time,
                       entry->second.end_time);

  entry->second.total_time += elapsed;
  entry->second.num_calls++;
  mutex_.unlock();
#endif
}

float Profiler::getTime(string id) {
  mutex_.lock();
  auto entry = profiler_.find(id);

  if (entry == profiler_.end()) {
    mutex_.unlock();
    return 0;
  }

  if(entry->second.num_calls == 0)
  {
    mutex_.unlock();
    return 0;
  }

  float elapsed =
      entry->second.total_time / static_cast<float>(entry->second.num_calls);
  mutex_.unlock();
  return elapsed;
}

float Profiler::getTimeGPU(string id)
{
#ifdef TRACKER_WITH_GPU
  mutex_.lock();
  auto entry = gpu_profiler_.find(id);

  if (entry == gpu_profiler_.end()) {
    mutex_.unlock();
    return 0;
  }

  if(entry->second.num_calls == 0)
  {
    mutex_.unlock();
    return 0;
  }

  float elapsed =
      entry->second.total_time / static_cast<float>(entry->second.num_calls);
  mutex_.unlock();
  return elapsed;
#endif
}

string Profiler::getProfile() {
  mutex_.lock();

  stringstream ss;
  //float overall = 0;
  for (auto it = profiler_.begin(); it != profiler_.end(); ++it) {
    float avg_time =
        it->second.total_time / static_cast<float>(it->second.num_calls);
    ss << it->first << ": " << avg_time << endl;
    //overall += avg_time;
  }
  #ifdef TRACKER_WITH_GPU
  for (auto it = gpu_profiler_.begin(); it != gpu_profiler_.end(); ++it) {
    float avg_time =
        it->second.total_time / static_cast<float>(it->second.num_calls);
    ss << it->first << " (gpu): " << avg_time << endl;
    //overall += avg_time;
  }
  #endif

  mutex_.unlock();
  //ss << "-----> overall: " << overall << " <-----\n";
  return ss.str();
}

}  // end namespace
