package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_library(
    name = "estimator_py",
    srcs = [
        "estimator_lib.py",
    ],
    srcs_version = "PY3",
    visibility = [
        "//tensorflow:__pkg__",
        "//tensorflow:internal",
    ],
    deps = [
        ":baseline",
        ":dnn",
        ":dnn_linear_combined",
        ":estimator",
        ":export",
        ":exporter",
        ":inputs",
        ":keras",
        ":linear",
        ":model_fn",
        ":parsing_utils",
        ":run_config",
        ":training",
        "//tensorflow:tensorflow_py_no_contrib",
    ],
)

py_library(
    name = "exporter",
    srcs = ["exporter.py"],
    srcs_version = "PY3",
    deps = [
        ":gc",
        ":metric_keys",
        ":util",
        "//tensorflow:tensorflow_py_no_contrib",
    ],
)

py_library(
    name = "gc",
    srcs = ["gc.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py_no_contrib",
    ],
)

py_library(
    name = "model_fn",
    srcs = ["model_fn.py"],
    srcs_version = "PY3",
    deps = [
        ":export_output",
        "//tensorflow:tensorflow_py_no_contrib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "training",
    srcs = ["training.py"],
    srcs_version = "PY3",
    deps = [
        ":estimator",
        ":exporter",
        ":run_config",
        "//tensorflow:tensorflow_py_no_contrib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "run_config",
    srcs = ["run_config.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py_no_contrib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "baseline",
    srcs = ["canned/baseline.py"],
    srcs_version = "PY3",
    deps = [
        ":estimator",
        ":head",
        ":model_fn",
        ":optimizers",
        "//tensorflow:tensorflow_py_no_contrib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "dnn",
    srcs = ["canned/dnn.py"],
    srcs_version = "PY3",
    deps = [
        ":estimator",
        ":head",
        ":model_fn",
        ":optimizers",
        "//tensorflow:tensorflow_py_no_contrib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "dnn_testing_utils",
    testonly = 1,
    srcs = ["canned/dnn_testing_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":estimator",
        ":head",
        ":metric_keys",
        ":model_fn",
        ":numpy_io",
        ":prediction_keys",
        "//tensorflow:tensorflow_py_no_contrib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
        "@six_archive//:six",
    ],
)

py_library(
    name = "dnn_linear_combined",
    srcs = ["canned/dnn_linear_combined.py"],
    srcs_version = "PY3",
    deps = [
        ":dnn",
        ":estimator",
        ":head",
        ":linear",
        ":model_fn",
        ":optimizers",
        "//tensorflow:tensorflow_py_no_contrib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "util",
    srcs = [
        "util.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py_no_contrib",
    ],
)

py_library(
    name = "estimator",
    srcs = [
        "estimator.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":export_export",
        ":model_fn",
        ":run_config",
        ":util",
        "//tensorflow:tensorflow_py_no_contrib",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "parsing_utils",
    srcs = [
        "canned/parsing_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py_no_contrib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "export_output",
    srcs = ["export/export_output.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py_no_contrib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "export",
    srcs = [
        "export/export_lib.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":export_export",
        ":export_output",
        "//tensorflow:tensorflow_py_no_contrib",
    ],
)

py_library(
    name = "export_export",
    srcs = [
        "export/export.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":util",
        "//tensorflow:tensorflow_py_no_contrib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "head",
    srcs = ["canned/head.py"],
    srcs_version = "PY3",
    deps = [
        ":export_output",
        ":metric_keys",
        ":model_fn",
        ":prediction_keys",
        "//tensorflow:tensorflow_py_no_contrib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "inputs",
    srcs = ["inputs/inputs.py"],
    srcs_version = "PY3",
    deps = [
        ":numpy_io",
        ":pandas_io",
        "//tensorflow:tensorflow_py_no_contrib",
    ],
)

py_library(
    name = "linear",
    srcs = ["canned/linear.py"],
    srcs_version = "PY3",
    deps = [
        ":estimator",
        ":head",
        ":optimizers",
        "//tensorflow:tensorflow_py_no_contrib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "linear_testing_utils",
    testonly = 1,
    srcs = ["canned/linear_testing_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":estimator",
        ":export_export",
        ":linear",
        ":metric_keys",
        ":numpy_io",
        ":pandas_io",
        ":run_config",
        "//tensorflow:tensorflow_py_no_contrib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "metric_keys",
    srcs = ["canned/metric_keys.py"],
    srcs_version = "PY3",
    deps = [
        ":model_fn",
    ],
)

py_library(
    name = "numpy_io",
    srcs = ["inputs/numpy_io.py"],
    srcs_version = "PY3",
    deps = [
        ":inputs_queues",
    ],
)

py_library(
    name = "optimizers",
    srcs = ["canned/optimizers.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py_no_contrib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "pandas_io",
    srcs = ["inputs/pandas_io.py"],
    srcs_version = "PY3",
    deps = [
        ":inputs_queues",
    ],
)

py_library(
    name = "prediction_keys",
    srcs = ["canned/prediction_keys.py"],
    srcs_version = "PY3",
)

py_library(
    name = "inputs_queues",
    srcs = [
        "inputs/queues/feeding_functions.py",
        "inputs/queues/feeding_queue_runner.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py_no_contrib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "keras",
    srcs = ["keras.py"],
    srcs_version = "PY3",
    deps = [
        ":estimator",
        ":export_export",
        ":model_fn",
        ":run_config",
        "//tensorflow:tensorflow_py_no_contrib",
    ],
)
