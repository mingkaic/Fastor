COPTS = select({
    "@bazel_tools//src/conditions:windows": ["/std:c++14"],
	"@bazel_tools//src/conditions:darwin": ["-std=c++14"],
	"//conditions:default": ["-std=c++14"],
})
