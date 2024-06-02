#include <iostream>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/types/optional.h"
#include "absl/status/status.h"

#include "cuda/cuda_init.h"

ABSL_FLAG(absl::optional<int>, device, absl::optional<int>(), "GPU device number.");

absl::Status find_cuda_device (void)
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
		return gpu_device_init(dev_id);
	}
	return gpu_device_init_max_gflops();
}

int main (int argc, char** argv)
{
	absl::ParseCommandLine(argc, argv);
	auto status = find_cuda_device();
	if (!status.ok())
	{
		std::cout << status << std::endl;
	}

	return 0;
}
