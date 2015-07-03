#ifndef GPU_TIMER_H
#define GPU_TIMER_H

#include <cuda_runtime.h>

struct GpuTimer
{
	cudaEvent_t m_start;
	cudaEvent_t m_stop;

	GpuTimer()
	{
		cudaEventCreate(&m_start);
		cudaEventCreate(&m_stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(m_start);
		cudaEventDestroy(m_stop);
	}

	void Start()
	{
		cudaEventRecord(m_start, 0);
	}

	void Stop()
	{
		cudaEventRecord(m_stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(m_stop);
		cudaEventElapsedTime(&elapsed, m_start, m_stop);
		return elapsed;
	}
};

#endif  /* GPU_TIMER_H__ */