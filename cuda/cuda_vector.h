#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"

#include "cuda/cuda_check.h"

#ifndef FASTOR_CUDA_CREATE_H
#define FASTOR_CUDA_CREATE_H

namespace cuda_helper
{

#ifdef __CUDA_RUNTIME_H__

inline absl::Status create_handle (cublasHandle_t& handle)
{
	CUBLAS_CHECK(cublasCreate(&handle));
	return absl::OkStatus();
}

template <typename T>
absl::StatusOr<T*> malloc (size_t n)
{
	T* ptr = nullptr;
	if (cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T)) != cudaSuccess)
	{
		auto msg = absl::StrFormat("GPU device memory allocation error (size=%d)", n);
		return absl::InternalError(msg);
	}
	return ptr;
}

template <typename T>
absl::Status set_vector (size_t n,
		T* loc, int loc_incr, T* gpu, int gpu_incr)
{
	if (cublasSetVector(n, sizeof(T), loc, loc_incr, gpu, gpu_incr) != CUBLAS_STATUS_SUCCESS)
	{
		return absl::InternalError("GPU device access error when setting data");
	}
	return absl::OkStatus();
}

template <typename T>
absl::Status sgemm (cublasHandle_t handle,
		cublasOperation_t transa, cublasOperation_t transb,
		int m, int n, int k,
		const T* alpha,
		const T* a, int lda,
		const T* b, int ldb,
		const T* beta,
		T* c, int ldc)
{
	if (cublasSgemm(handle, transa, transb, m, n, k,
				alpha, a, lda, b, ldb, beta, c, ldc) != CUBLAS_STATUS_SUCCESS)
	{
		return absl::InternalError("Sgemm kernel execution error.");
	}
	return absl::OkStatus();
}

#endif // __CUDA_RUNTIME_H__

}

#endif // FASTOR_CUDA_CREATE_H
