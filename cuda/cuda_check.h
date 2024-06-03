#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "absl/status/status.h"

#ifndef FASTOR_CUDA_CHECK_H
#define FASTOR_CUDA_CHECK_H

namespace cuda_helper
{

#ifdef __CUDA_RUNTIME_H__

static inline absl::Status cublas_to_status (
		const cublasStatus_t& err, const char* filename, int line)
{
		if (err != CUBLAS_STATUS_SUCCESS)
		{
			auto msg = absl::StrFormat("CUBLAS Error: %d at %s:%d\n", err, filename, line);
			return absl::InternalError(msg);
		}
		return absl::OkStatus();
}

static inline absl::Status cublas_to_status (
		const cudaError_t& err, const char* filename, int line)
{
		if (err != cudaSuccess)
		{
			auto msg = absl::StrFormat("CUBLAS Error: %d at %s:%d\n", err, filename, line);
			return absl::InternalError(msg);
		}
		return absl::OkStatus();
}

#define CUBLAS_CHECK(expr)                                        \
	do                                                            \
	{                                                             \
		auto status = cublas_to_status(expr, __FILE__, __LINE__); \
		if (!status.ok()) return status;                          \
	} while (0)

#endif // __CUDA_RUNTIME_H__

}

#endif // FASTOR_CUDA_CHECK_H
