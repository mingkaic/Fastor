#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"

#ifndef FASTOR_CUDA_INIT_H
#define FASTOR_CUDA_INIT_H

struct SMtoArch
{
	int sm_;
	const char* name_;
	int cores_;
};

static const char* cuda_no_devices_err = "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n";

static const char* cuda_device_err_fmt = R"(
>> %d CUDA capable GPU device(s) detected. <<
>> gpuDeviceInit (-device=%d) is not a valid GPU device. <<

)";

static const char* cuda_device_prohibited_err =
	"Error: device is running in <Compute Mode Prohibited>, no threads can use cudaSetDevice().\n";

static const char* cuda_not_supported = "Error: GPU device does not support CUDA.\n";

static const SMtoArch gpu_archs[] = {
	{0x30, "Kepler", 192},
	{0x32, "Kepler", 192},
	{0x35, "Kepler", 192},
	{0x37, "Kepler", 192},
	{0x50, "Maxwell", 128},
	{0x52, "Maxwell", 128},
	{0x53, "Maxwell", 128},
	{0x60, "Pascal", 64},
	{0x61, "Pascal", 128},
	{0x62, "Pascal", 128},
	{0x70, "Volta", 64},
	{0x72, "Xavier", 64},
	{0x75, "Turing", 64},
	{0x80, "Ampere", 64},
	{0x86, "Ampere", 128},
	{0x87, "Ampere", 128},
	{0x89, "Ada", 128},
	{0x90, "Hopper", 128},
	{-1, "Graphics Device", 1},
};

static inline const SMtoArch& find_arch (int major, int minor)
{
	int i;
	for (i = 0; gpu_archs[i].sm_ != -1; ++i)
	{
		if (gpu_archs[i].sm_ == ((major << 4) + minor))
		{
			return gpu_archs[i];
		}
	}
	// if we don't find the correct version, use the default version.
	printf("sm %d.%d is undefined. Default to use %s; %d Cores/SM\n",
			major, minor, gpu_archs[i].name_, gpu_archs[i].cores_);
	return gpu_archs[i];
}

static inline const char* smver_to_archname (int major, int minor)
{
	return find_arch(major, minor).name_;
}

static inline int smver_to_cores (int major, int minor)
{
	return find_arch(major, minor).cores_;
}

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

inline absl::Status gpu_device_init (int dev_id)
{
	int device_count = 0;
	CUBLAS_CHECK(cudaGetDeviceCount(&device_count));

	if (device_count == 0)
	{
		return absl::InternalError(cuda_no_devices_err);
	}

	if (dev_id < 0)
	{
		dev_id = 0;
	}

	if (dev_id > device_count - 1)
	{
		auto msg = absl::StrFormat(cuda_device_err_fmt, device_count, dev_id);
		return absl::InvalidArgumentError(msg);
	}

	int compute_mode = -1, major = 0, minor = 0;
	CUBLAS_CHECK(cudaDeviceGetAttribute(&compute_mode, cudaDevAttrComputeMode, dev_id));
	CUBLAS_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev_id));
	CUBLAS_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev_id));
	if (compute_mode == cudaComputeModeProhibited)
	{
		return absl::InternalError(cuda_device_prohibited_err);
	}

	if (major < 1)
	{
		return absl::InternalError(cuda_not_supported);
	}

	CUBLAS_CHECK(cudaSetDevice(dev_id));
	printf("gpu_device_init() CUDA Device [%d]: \"%s\"\n", dev_id, smver_to_archname(major, minor));

	return absl::OkStatus();
}

inline absl::StatusOr<int> get_gpu_max_gflops_dev_id (void)
{
	int sm_per_multiproc = 0;
	int max_perf_dev = 0;
	int devices_prohibited = 0;

	uint64_t max_compute_perf = 0;

	int device_count = 0;
	CUBLAS_CHECK(cudaGetDeviceCount(&device_count));

	if (device_count == 0)
	{
		return absl::InternalError(cuda_no_devices_err);
	}

	for (int current_dev = 0; current_dev < device_count; ++current_dev)
	{
		int compute_mode = -1, major = 0, minor = 0;
		CUBLAS_CHECK(cudaDeviceGetAttribute(&compute_mode, cudaDevAttrComputeMode, current_dev));
		CUBLAS_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, current_dev));
		CUBLAS_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, current_dev));

		if (compute_mode == cudaComputeModeProhibited)
		{
			++devices_prohibited;
			continue;
		}
		if (major == 9999 && minor == 9999)
		{
			sm_per_multiproc = 1;
		}
		else
		{
			sm_per_multiproc = smver_to_cores(major, minor);
		}
		int multi_processor_count = 0, clock_rate = 0;
		CUBLAS_CHECK(cudaDeviceGetAttribute(
					&multi_processor_count, cudaDevAttrMultiProcessorCount, current_dev));
		cudaError_t result = cudaDeviceGetAttribute(
				&clock_rate, cudaDevAttrClockRate, current_dev);
		if (result != cudaSuccess)
		{
			if (result == cudaErrorInvalidValue)
			{
				clock_rate = 1;
			}
			else
			{
				auto msg = absl::StrFormat("CUDA error at %s:%d code=%d(%s)\n", __FILE__, __LINE__,
						static_cast<unsigned int>(result), cudaGetErrorName(result));
				return absl::InternalError(msg);
			}
		}
		uint64_t compute_perf = (uint64_t) multi_processor_count * sm_per_multiproc * clock_rate;
		if (compute_perf > max_compute_perf)
		{
			max_compute_perf = compute_perf;
			max_perf_dev = current_dev;
		}
	}

	if (devices_prohibited == device_count)
	{
		return absl::InternalError("Error: all devices have compute mode prohibited.\n");
	}
	return max_perf_dev;
}

inline absl::Status gpu_device_init_max_gflops (void)
{
	auto result = get_gpu_max_gflops_dev_id();
	if (result.ok())
	{
		int dev_id = result.value();
		CUBLAS_CHECK(cudaSetDevice(dev_id));
		int major = 0, minor = 0;
		CUBLAS_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev_id));
		CUBLAS_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev_id));
		printf("gpu_device_init() CUDA Device [%d]: \"%s\"\n",
				dev_id, smver_to_archname(major, minor));
	}
	return result.status();
}

#endif // __CUDA_RUNTIME_H__

#endif // FASTOR_CUDA_INIT_H
