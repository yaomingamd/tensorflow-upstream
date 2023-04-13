# Change Log for ROCm tensorflow

This is a template changelog to record changes to tensorflow staging branches.
This CHANGELOG will only indicate features that were staged to the testing branch during the corresponding ROCm release process.

## TensorFlow for ROCm, r2.12-rocm-enhanced

### Added
*   Changes to track call-context information [15dc6beea5d46da075bc52088351232756b04a80](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/b056bd88b4a624c0deb9260a66146ba5cdd8903b)
*   Fix unused and ignored status errors [28b1e74b11e77322ca3869e7af7435bf19462210](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/28b1e74b11e77322ca3869e7af7435bf19462210)
*   Avoid runtime errors with enhanced call context when mlir bridge is enabled  [b056bd88b4a624c0deb9260a66146ba5cdd8903b](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/b056bd88b4a624c0deb9260a66146ba5cdd8903b)
*   Adjust mlir bridge check [6aecd729e555b89b34127a88131c3fbd1b7edada](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/6aecd729e555b89b34127a88131c3fbd1b7edada)
*   Only send call context for GPUs [05b7a00599e26081394a6e31961bd2398bd06456](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/05b7a00599e26081394a6e31961bd2398bd06456)
*   Fixes for call context feature [c13d87e3c4f5afc8ac907db30fc2644cad39a4fa](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/c13d87e3c4f5afc8ac907db30fc2644cad39a4fa)
*   [ROCm] Include rocm_config.h in rocm_dnn [0de6f941383f5c884c1a1bf2614a89ad1d2c2150](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/0de6f941383f5c884c1a1bf2614a89ad1d2c2150)
*   Add py3.11 to setup.build-python.sh [7459ffb15aabc6e0ead60400d195bd12c653ffb8](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/7459ffb15aabc6e0ead60400d195bd12c653ffb8)
*   Revert: Migrate xla_test macro to use xla_cc_test instead of tf_cc_test[034d37f2b42a378fe61cd3eaabc0689dbeb87749](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/034d37f2b42a378fe61cd3eaabc0689dbeb87749)
*   fixed related matmul tests due to undefined attribute grad_a[50b2f03319c79219dd8fc456e3eb164051c73563](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/50b2f03319c79219dd8fc456e3eb164051c73563)
*   Use tf_cc_binary for hlo_to_llvm_ir[6d0df5962a3f4001774e5aebdc6909892a384300](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/6d0df5962a3f4001774e5aebdc6909892a384300)
*   Use tf_cc_test for a few more tests in 2.12[667168052d044b4dfebef740b5ef51fd32ff596a](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/667168052d044b4dfebef740b5ef51fd32ff596a)
*   Fix pkg versioning when in RC[45cff9783da5fec4eeae901760fd17260f725423](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/45cff9783da5fec4eeae901760fd17260f725423)
*   pin keras, tb, estimator in tensorflow-build for 2.12 release[01b692523080cb761539bc36454754d4bce42f1e](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/01b692523080cb761539bc36454754d4bce42f1e)
*   Set TF_TESTS_PER_GPU=1 in gpu.bazelrc[405af7f760b33d99f4c84f05f7460d1a41a14979](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/405af7f760b33d99f4c84f05f7460d1a41a14979)
*   Add new runtime image for TF[0bdf723810fd3500c2abb4ee022bce2924d669a5](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/0bdf723810fd3500c2abb4ee022bce2924d669a5)
*   Add some extra pkgs to tensorflow-build so it build python properly[2428f7d7f9340af10c6ceeb5f52d42f5e05ba031](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/2428f7d7f9340af10c6ceeb5f52d42f5e05ba031)
*   fix base docker build for internal ci[67f74e3fb00b5b0880bd31c4822830aad7c7c826](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/67f74e3fb00b5b0880bd31c4822830aad7c7c826)
*   Update setup.rocm.cs7.sh[f0a51d861dbe42181b4db2f54d654ff76535f182](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/f0a51d861dbe42181b4db2f54d654ff76535f182)
*   Use CUSTOM_INSTALL as the variable name of the installation script[75a5a35d83ee93f8f1e5950a959b55d2d60e7dd3](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/75a5a35d83ee93f8f1e5950a959b55d2d60e7dd3)
*   Add readline-devel to tf-build image[28b080d5e54158f2fbf92c882f3a3b33175b2f8e](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/28b080d5e54158f2fbf92c882f3a3b33175b2f8e)
*   [ROCM] Fix includes for ROCm 5.6[62ff1dceb232939251611603bd90f06775b634c5](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/62ff1dceb232939251611603bd90f06775b634c5)
*   Fix the location of tensorflow wheel package in the runtime Dockerfile[66634df02e1ccad037d0538cc9d3695e9cd83802](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/66634df02e1ccad037d0538cc9d3695e9cd83802)
*   add more linux distro support for runtime dockerfile[503951188ad8e0116c0bab0b2ff40dba39f4465e](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/503951188ad8e0116c0bab0b2ff40dba39f4465e)
*   Update golden api, enable api_compatibility_test[b98db55a91643112be90ad244c0931b023fb2921](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/b98db55a91643112be90ad244c0931b023fb2921)
*   Fix python3 set up in Dockerfile.rocm[29facd1cda73c6b35d7cc3b6aa6bf29290f786ee](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/29facd1cda73c6b35d7cc3b6aa6bf29290f786ee)
*   Update golden api for context tracking update[d95f83da36315f28595d61309af029fe15c007a0](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/d95f83da36315f28595d61309af029fe15c007a0)
*   Fix gradient_input_output_exclusions_test[8cfe329c249e7786d9f116a3ed627f4a11b96d4e](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/8cfe329c249e7786d9f116a3ed627f4a11b96d4e)
*   More Replace xla_cc_test/binary with tf_cc_test/binary (cpu tests)[baa5e035c523ce452f2aeb3943968d8f74fd6f6e](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/baa5e035c523ce452f2aeb3943968d8f74fd6f6e)


## Upstream TensorFlow release notes

# Release 2.12.0

## TensorFlow

### Breaking Changes

*   Build, Compilation and Packaging

    *   Removed redundant packages `tensorflow-gpu` and `tf-nightly-gpu`. These packages were removed and replaced with packages that direct users to switch to `tensorflow` or `tf-nightly` respectively. Since TensorFlow 2.1, the only difference between these two sets of packages was their names, so there is no loss of functionality or GPU support. See https://pypi.org/project/tensorflow-gpu for more details.

*   `tf.function`:

    *   `tf.function` now uses the Python inspect library directly for parsing the signature of the Python function it is decorated on. This change may break code where the function signature is malformed, but was ignored previously, such as:
        *   Using `functools.wraps` on a function with different signature
        *   Using `functools.partial` with an invalid `tf.function` input
    *   `tf.function` now enforces input parameter names to be valid Python identifiers. Incompatible names are automatically sanitized similarly to existing SavedModel signature behavior.
    *   Parameterless `tf.function`s are assumed to have an empty `input_signature` instead of an undefined one even if the `input_signature` is unspecified.
    *   `tf.types.experimental.TraceType` now requires an additional `placeholder_value` method to be defined.
    *   `tf.function` now traces with placeholder values generated by TraceType instead of the value itself.

*   Experimental APIs `tf.config.experimental.enable_mlir_graph_optimization` and `tf.config.experimental.disable_mlir_graph_optimization` were removed.

### Major Features and Improvements

*  Support for Python 3.11 has been added.
*  Support for Python 3.7 has been removed. We are not releasing any more patches for Python 3.7.

*   `tf.lite`:

    *   Add 16-bit float type support for built-in op `fill`.
    *   Transpose now supports 6D tensors.
    *   Float LSTM now supports diagonal recurrent tensors: https://arxiv.org/abs/1903.08023

*   `tf.experimental.dtensor`:

    *   Coordination service now works with `dtensor.initialize_accelerator_system`, and enabled by default.
    *   Add `tf.experimental.dtensor.is_dtensor` to check if a tensor is a DTensor instance.

*   `tf.data`:

    *   Added support for alternative checkpointing protocol which makes it possible to checkpoint the state of the input pipeline without having to store the contents of internal buffers. The new functionality can be enabled through the `experimental_symbolic_checkpoint` option of `tf.data.Options()`.
    *   Added a new `rerandomize_each_iteration` argument for the `tf.data.Dataset.random()` operation, which controls whether the sequence of generated random numbers should be re-randomized every epoch or not (the default behavior). If `seed` is set and `rerandomize_each_iteration=True`, the `random()` operation will produce a different (deterministic) sequence of numbers every epoch.
    *   Added a new `rerandomize_each_iteration` argument for the `tf.data.Dataset.sample_from_datasets()` operation, which controls whether the sequence of generated random numbers used for sampling should be re-randomized every epoch or not. If `seed` is set and `rerandomize_each_iteration=True`, the `sample_from_datasets()` operation will use a different (deterministic) sequence of numbers every epoch.

*   `tf.test`:

    *   Added `tf.test.experimental.sync_devices`, which is useful for accurately measuring performance in benchmarks.

*   `tf.experimental.dtensor`:

    *   Added experimental support to ReduceScatter fuse on GPU (NCCL).

### Bug Fixes and Other Changes

*   `tf.SavedModel`:
    * Introduced new class `tf.saved_model.experimental.Fingerprint` that contains the fingerprint of the SavedModel. See the [SavedModel Fingerprinting RFC](https://github.com/tensorflow/community/pull/415) for details.
    * Introduced API `tf.saved_model.experimental.read_fingerprint(export_dir)` for reading the fingerprint of a SavedModel.
* `tf.random`
  * Added non-experimental aliases for `tf.random.split` and `tf.random.fold_in`, the experimental endpoints are still available so no code changes are necessary.
* `tf.experimental.ExtensionType`
  * Added function `experimental.extension_type.as_dict()`, which converts an instance of `tf.experimental.ExtensionType` to a `dict` representation.
* `stream_executor`
  * Top level `stream_executor` directory has been deleted, users should use equivalent headers and targets under `compiler/xla/stream_executor`.
* `tf.nn`
  * Added `tf.nn.experimental.general_dropout`, which is similar to `tf.random.experimental.stateless_dropout` but accepts a custom sampler function.
* `tf.types.experimental.GenericFunction`
  * The `experimental_get_compiler_ir` method supports tf.TensorSpec compilation arguments.
*  `tf.config.experimental.mlir_bridge_rollout`
    *   Removed enums `MLIR_BRIDGE_ROLLOUT_SAFE_MODE_ENABLED` and `MLIR_BRIDGE_ROLLOUT_SAFE_MODE_FALLBACK_ENABLED` which are no longer used by the tf2xla bridge

## Keras

 Keras is a framework built on top of the TensorFlow. See more details on the Keras [website](https://keras.io/).

### Breaking Changes


`tf.keras`:

* Moved all saving-related utilities to a new namespace, `keras.saving`, for example: `keras.saving.load_model`, `keras.saving.save_model`, `keras.saving.custom_object_scope`, `keras.saving.get_custom_objects`, `keras.saving.register_keras_serializable`,`keras.saving.get_registered_name` and `keras.saving.get_registered_object`. The previous API locations (in `keras.utils` and `keras.models`) will be available indefinitely, but we recommend you update your code to point to the new API locations.
 * Improvements and fixes in Keras loss masking:
    * Whether you represent a ragged tensor as a `tf.RaggedTensor` or using [keras masking](https://www.tensorflow.org/guide/keras/masking_and_padding), the returned loss values should be the identical to each other. In previous versions Keras may have silently ignored the mask.
 * If you use masked losses with Keras the loss values may be different in TensorFlow `2.12` compared to previous versions.
 * In cases where the mask was previously ignored, you will now get an error if you pass a mask with an incompatible shape.

### Major Features and Improvements     

`tf.keras`:

 *   The new Keras model saving format (`.keras`) is available. You can start using it via `model.save(f"{fname}.keras", save_format="keras_v3")`. In the future it will become the default for all files with the `.keras` extension. This file format targets the Python runtime only and makes it possible to reload Python objects identical to the saved originals. The format supports non-numerical state such as vocabulary files and lookup tables, and it is easy to customize in the case of custom layers with exotic elements of state (e.g. a FIFOQueue). The format does not rely on bytecode or pickling, and is safe by default. Note that as a result, Python `lambdas` are disallowed at loading time. If you want to use `lambdas`, you can pass `safe_mode=False` to the loading method (only do this if you trust the source of the model).
*   Added a `model.export(filepath)` API to create a lightweight SavedModel artifact that can be used for inference (e.g. with TF-Serving).
*   Added `keras.export.ExportArchive` class for low-level customization of the process of exporting SavedModel artifacts for inference. Both ways of exporting models are based on `tf.function` tracing and produce a TF program composed of TF ops. They are meant primarily for environments where the TF runtime is available, but not the Python interpreter, as is typical for production with TF Serving.
 *   Added utility `tf.keras.utils.FeatureSpace`, a one-stop shop for structured data preprocessing and encoding.
 *   Added `tf.SparseTensor` input support to `tf.keras.layers.Embedding` layer. The layer now accepts a new boolean argument `sparse`. If `sparse` is set to True, the layer returns a SparseTensor instead of a dense Tensor. Defaults to False.
 *   Added `jit_compile` as a settable property to `tf.keras.Model`.
 *   Added `synchronized` optional parameter to `layers.BatchNormalization`.
 *   Added deprecation warning to `layers.experimental.SyncBatchNormalization` and suggested to use `layers.BatchNormalization` with `synchronized=True` instead.
 *   Updated `tf.keras.layers.BatchNormalization` to support masking of the inputs (`mask` argument) when computing the mean and variance.
 *   Add `tf.keras.layers.Identity`, a placeholder pass-through layer.
 *   Add `show_trainable` option to `tf.keras.utils.model_to_dot` to display layer trainable status in model plots.
 *   Add ability to save a `tf.keras.utils.FeatureSpace` object, via `feature_space.save("myfeaturespace.keras")`, and reload it via `feature_space = tf.keras.models.load_model("myfeaturespace.keras")`.
*   Added utility `tf.keras.utils.to_ordinal` to convert class vector to ordinal regression / classification matrix.

### Bug Fixes and Other Changes

*   N/A

## Security

*   Fixes an FPE in TFLite in conv kernel [CVE-2023-27579](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2023-27579)
*   Fixes a double free in Fractional(Max/Avg)Pool [CVE-2023-25801](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2023-25801)
*   Fixes a null dereference on ParallelConcat with XLA [CVE-2023-25676](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2023-25676)
*   Fixes a segfault in Bincount with XLA [CVE-2023-25675](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2023-25675)
*   Fixes an NPE in RandomShuffle with XLA enable [CVE-2023-25674](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2023-25674)
*   Fixes an FPE in TensorListSplit with XLA [CVE-2023-25673](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2023-25673)
*   Fixes segmentation fault in tfg-translate [CVE-2023-25671](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2023-25671)
*   Fixes an NPE in QuantizedMatMulWithBiasAndDequantize [CVE-2023-25670](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2023-25670)
*   Fixes an FPE in AvgPoolGrad with XLA [CVE-2023-25669](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2023-25669)
*   Fixes a heap out-of-buffer read vulnerability in the QuantizeAndDequantize operation [CVE-2023-25668](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2023-25668)
*   Fixes a segfault when opening multiframe gif [CVE-2023-25667](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2023-25667)
*   Fixes an NPE in SparseSparseMaximum [CVE-2023-25665](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2023-25665)
*   Fixes an FPE in AudioSpectrogram [CVE-2023-25666](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2023-25666)
*   Fixes a heap-buffer-overflow in AvgPoolGrad  [CVE-2023-25664](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2023-25664)
*   Fixes a NPE in TensorArrayConcatV2  [CVE-2023-25663](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2023-25663)
*   Fixes a Integer overflow in EditDistance  [CVE-2023-25662](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2023-25662)
*   Fixes a Seg fault in `tf.raw_ops.Print` [CVE-2023-25660](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2023-25660)
*   Fixes a OOB read in DynamicStitch [CVE-2023-25659](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2023-25659)
*   Fixes a OOB Read in GRUBlockCellGrad [CVE-2023-25658](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2023-25658)

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

103yiran, 8bitmp3, Aakar, Aakar Dwivedi, Abinash Satapathy, Aditya Kane, ag.ramesh, Alexander Grund, Andrei Pikas, andreii, Andrew Goodbody, angerson, Anthony_256, Ashay Rane, Ashiq Imran, Awsaf, Balint Cristian, Banikumar Maiti (Intel Aipg), Ben Barsdell, bhack, cfRod, Chao Chen, chenchongsong, Chris Mc, Daniil Kutz, David Rubinstein, dianjiaogit, dixr, Dongfeng Yu, dongfengy, drah, Eric Kunze, Feiyue Chen, Frederic Bastien, Gauri1 Deshpande, guozhong.zhuang, hDn248, HYChou, ingkarat, James Hilliard, Jason Furmanek, Jaya, Jens Glaser, Jerry Ge, Jiao Dian'S Power Plant, Jie Fu, Jinzhe Zeng, Jukyy, Kaixi Hou, Kanvi Khanna, Karel Ha, karllessard, Koan-Sin Tan, Konstantin Beluchenko, Kulin Seth, Kun Lu, Kyle Gerard Felker, Leopold Cambier, Lianmin Zheng, linlifan, liuyuanqiang, Lukas Geiger, Luke Hutton, Mahmoud Abuzaina, Manas Mohanty, Mateo Fidabel, Maxiwell S. Garcia, Mayank Raunak, mdfaijul, meatybobby, Meenakshi Venkataraman, Michael Holman, Nathan John Sircombe, Nathan Luehr, nitins17, Om Thakkar, Patrice Vignola, Pavani Majety, per1234, Philipp Hack, pollfly, Prianka Liz Kariat, Rahul Batra, rahulbatra85, ratnam.parikh, Rickard Hallerbäck, Roger Iyengar, Rohit Santhanam, Roman Baranchuk, Sachin Muradi, sanadani, Saoirse Stewart, seanshpark, Shawn Wang, shuw, Srinivasan Narayanamoorthy, Stewart Miles, Sunita Nadampalli, SuryanarayanaY, Takahashi Shuuji, Tatwai Chong, Thibaut Goetghebuer-Planchon, tilakrayal, Tirumalesh, TJ, Tony Sung, Trevor Morris, unda, Vertexwahn, Vinila S, William Muir, Xavier Bonaventura, xiang.zhang, Xiao-Yong Jin, yleeeee, Yong Tang, Yuriy Chernyshov, Zhang, Xiangze, zhaozheng09

