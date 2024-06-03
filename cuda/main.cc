#include <iostream>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/types/optional.h"
#include "absl/status/status.h"

#include "cuda/cuda_init.h"
#include "cuda/cuda_vector.h"

ABSL_FLAG(absl::optional<int>, device, absl::optional<int>(), "GPU device number.");

absl::Status init_cuda_device (void)
{
	int dev_id = 0;
	auto d = absl::GetFlag(FLAGS_device);
	if (d.has_value())
	{
		dev_id = d.value();
		if (dev_id < 0)
		{
			return absl::InvalidArgumentError("Invalid command line parameter\n");
		}
		return cuda_helper::gpu_device_init(dev_id);
	}
	return cuda_helper::gpu_device_init_max_gflops().status();
}

static void simple_sgemm (int n, float alpha, const float *A, const float *B,
		float beta, float *C)
{
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			float prod = 0;
			for (int k = 0; k < n; ++k)
			{
				prod += A[k * n + i] * B[j * n + k];
			}
			C[j * n + i] = alpha * prod + beta * C[j * n + i];
		}
	}
}

#define STATUS_CHECK(XPR) {\
	auto status = (XPR);\
	if (!status.ok())\
	{\
		std::cerr << status << std::endl;\
		return EXIT_FAILURE;\
	}\
}

#define STATUSOR_CHECK(TYPE, VAR, XPR) \
	TYPE VAR; {\
	auto status_or = (XPR);\
	if (!status_or.ok())\
	{\
		std::cerr << status_or.status() << std::endl;\
		return EXIT_FAILURE;\
	}\
	VAR = status_or.value();\
}

int main (int argc, char** argv)
{
	absl::ParseCommandLine(argc, argv);
	STATUS_CHECK(init_cuda_device());

	// https://github.com/NVIDIA/cuda-samples/blob/master/Samples/4_CUDA_Libraries/simpleCUBLAS/simpleCUBLAS.cpp
	cublasHandle_t handle;
	STATUS_CHECK(cuda_helper::create_handle(handle));

	/* Allocate host memory for the matrices */
	const int n = 275;
	const int n2 = n * n;

	float h_A[n2];
	float h_B[n2];
	float h_C[n2];

	/* Fill the matrices with test data */
	for (size_t i = 0; i < n2; i++)
	{
		h_A[i] = rand() / static_cast<float>(RAND_MAX);
		h_B[i] = rand() / static_cast<float>(RAND_MAX);
		h_C[i] = rand() / static_cast<float>(RAND_MAX);
	}

	/* Allocate device memory for the matrices */
	STATUSOR_CHECK(float*, d_A, cuda_helper::malloc<float>(n2))
	STATUSOR_CHECK(float*, d_B, cuda_helper::malloc<float>(n2))
	STATUSOR_CHECK(float*, d_C, cuda_helper::malloc<float>(n2))

	/* Initialize the device matrices with the host matrices */
	STATUS_CHECK(cuda_helper::set_vector(n2, h_A, 1, d_A, 1))
	STATUS_CHECK(cuda_helper::set_vector(n2, h_B, 1, d_B, 1))
	STATUS_CHECK(cuda_helper::set_vector(n2, h_C, 1, d_C, 1))

	float alpha = 1.0f;
	float beta = 0.0f;

	/* Performs operation using plain C code */
	simple_sgemm(n, alpha, h_A, h_B, beta, h_C);

	/* Performs operation using cublas */
	STATUS_CHECK(cuda_helper::sgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
				n, n, n, &alpha, d_A, n, d_B, n, &beta, d_C, n))

	/* Allocate host memory for reading back the result from device memory */
	float* h2_C = new float[n2];
	if (h2_C == 0)
	{
		std::cerr << "!!!! host memory allocation error (C)" << std::endl;
		return EXIT_FAILURE;
	}

	/* Read the result back */
	if (cublasGetVector(n2, sizeof(h2_C[0]), d_C, 1, h2_C, 1) != CUBLAS_STATUS_SUCCESS)
	{
		std::cerr << "!!!! device access error (read C)" << std::endl;
		return EXIT_FAILURE;
	}

	/* Check result against reference */
	double error_norm = 0;
	double ref_norm = 0;
	double diff;

	for (size_t i = 0; i < n2; ++i)
	{
		diff = h_C[i] - h2_C[i];
		error_norm += diff * diff;
		ref_norm += h2_C[i] * h2_C[i];
	}

	error_norm = std::sqrt(error_norm);
	ref_norm = std::sqrt(ref_norm);

	if (std::abs(ref_norm) < 1e-7)
	{
		std::cerr << "!!!! reference norm is 0" << std::endl;
		return EXIT_FAILURE;
	}

	/* Memory clean up */
	delete[] h2_C;

	if (cudaFree(d_A) != cudaSuccess)
	{
		std::cerr << "!!!! memory free error (A)" << std::endl;
		return EXIT_FAILURE;
	}

	if (cudaFree(d_B) != cudaSuccess)
	{
		std::cerr << "!!!! memory free error (B)" << std::endl;
		return EXIT_FAILURE;
	}

	if (cudaFree(d_C) != cudaSuccess)
	{
		std::cerr << "!!!! memory free error (C)" << std::endl;
		return EXIT_FAILURE;
	}

	/* Shutdown */
	if (cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS)
	{
		std::cerr << "!!!! shutdown error (A)" << std::endl;
		return EXIT_FAILURE;
	}

	/* Final report */
	if (error_norm / ref_norm < 1e-6f)
	{
		std::cout << "simpleCUBLAS test passed (" << error_norm << ")." << std::endl;
	}
	else
	{
		std::cerr << "simpleCUBLAS test failed (" << error_norm << ")." << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
