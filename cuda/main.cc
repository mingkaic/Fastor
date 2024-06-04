#include <iostream>

#include <Fastor/Fastor.h>

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

static void simple_sgemm (int m, int n, int k, float alpha, const float *A, const float *B,
		float beta, float *C)
{
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			float prod = 0;
			for (int l = 0; l < k; ++l)
			{
				prod += A[l * m + i] * B[j * k + l];
			}
			C[j * m + i] = alpha * prod + beta * C[j * m + i];
		}
	}
}

template <int M, int N, int K>
static void simple_sgemm2 (float *A, float *B, float *C)
{
	Fastor::TensorMap<float,M,K> a(A);
	Fastor::TensorMap<float,K,N> b(B);
	Fastor::TensorMap<float,M,N> c(C);
	c = Fastor::einsum<Fastor::Index<0,1>,Fastor::Index<1,2>>(a, b);
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
	const int m = 277;
	const int n = 276;
	const int k = 275;
	const int nA = m * k;
	const int nB = k * n;
	const int nC = m * n;

	float h_A[nA];
	float h_B[nB];
	float h_C[nC];

	float* h2_C = new float[nC];
	float* h3_C = new float[nC];

	/* Allocate host memory for reading back the result from device memory */
	if (h2_C == 0)
	{
		std::cerr << "!!!! host memory allocation error (C)" << std::endl;
		return EXIT_FAILURE;
	}

	/* Fill the matrices with test data */
	for (size_t i = 0; i < nA; i++)
	{
		h_A[i] = rand() / static_cast<float>(RAND_MAX);
	}
	for (size_t i = 0; i < nB; i++)
	{
		h_B[i] = rand() / static_cast<float>(RAND_MAX);
	}
	std::fill(h_C, h_C + nC, 0);
	std::fill(h3_C, h3_C + nC, 0);

	/* Allocate device memory for the matrices */
	STATUSOR_CHECK(float*, d_A, cuda_helper::malloc<float>(nA))
	STATUSOR_CHECK(float*, d_B, cuda_helper::malloc<float>(nB))
	STATUSOR_CHECK(float*, d_C, cuda_helper::malloc<float>(nC))

	/* Initialize the device matrices with the host matrices */
	STATUS_CHECK(cuda_helper::set_vector(nA, h_A, 1, d_A, 1))
	STATUS_CHECK(cuda_helper::set_vector(nB, h_B, 1, d_B, 1))
	STATUS_CHECK(cuda_helper::set_vector(nC, h_C, 1, d_C, 1))

	float alpha = 1.0f;
	float beta = 0.0f;

	/* Performs operation using plain C code */
	simple_sgemm(m, n, k, alpha, h_A, h_B, beta, h_C);

	//simple_sgemm2<m, n, k>(h_A, h_B, h3_C);

	/* Performs operation using cublas */
	STATUS_CHECK(cuda_helper::sgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
				m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m))

	/* Read the result back */
	if (cublasGetVector(nC, sizeof(h2_C[0]), d_C, 1, h2_C, 1) != CUBLAS_STATUS_SUCCESS)
	{
		std::cerr << "!!!! device access error (read C)" << std::endl;
		return EXIT_FAILURE;
	}

	/* Check result against reference */
	double error_norm = 0;
	double ref_norm = 0;
	double error_norm2 = 0;
	double ref_norm2 = 0;
	double diff;

	for (size_t i = 0; i < nC; ++i)
	{
		diff = h_C[i] - h2_C[i];
		error_norm += diff * diff;
		ref_norm += h2_C[i] * h2_C[i];

		diff = h_C[i] - h3_C[i];
		error_norm2 += diff * diff;
		ref_norm2 += h3_C[i] * h3_C[i];
	}

	error_norm = std::sqrt(error_norm);
	ref_norm = std::sqrt(ref_norm);

	error_norm2 = std::sqrt(error_norm2);
	ref_norm2 = std::sqrt(ref_norm2);

	if (std::abs(ref_norm) < 1e-7)
	{
		std::cerr << "!!!! reference norm is 0" << std::endl;
		return EXIT_FAILURE;
	}

	/* Memory clean up */
	delete[] h2_C;
	delete[] h3_C;

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
		std::cout << "simpleCUBLAS test alternative err (" << error_norm2 << "), ref (" << ref_norm2 << ")." << std::endl;
	}
	else
	{
		std::cerr << "simpleCUBLAS test failed (" << error_norm << ")." << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
