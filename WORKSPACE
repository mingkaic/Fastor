workspace(name = "com_github_romeric_fastor")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "rules_cuda",
    sha256 = "5833421DC605F1E034F877F92A7971FA7D3BB9DBB53E8AE1AE3AE1A3512FB09E",
    strip_prefix = "rules_cuda-b4f909345d6455793b462de2e0079fb375664972",
    urls = ["https://github.com/bazel-contrib/rules_cuda/archive/b4f909345d6455793b462de2e0079fb375664972.tar.gz"],
)
load("@rules_cuda//cuda:repositories.bzl", "register_detected_cuda_toolchains", "rules_cuda_dependencies")
rules_cuda_dependencies()
register_detected_cuda_toolchains()
