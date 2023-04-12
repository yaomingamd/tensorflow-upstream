# Change Log for ROCm tensorflow

This is a template changelog to record changes to tensorflow staging branches.
This CHANGELOG will only indicate features that were staged to the testing branch during the corresponding ROCm release process.

The develop-upstream-QA-branch was branched at commit d85b57a4205d5902676c00ce7edde9d20bcabfe8

## TensorFlow for ROCm, develop-upstream-QA-rocm56

### Added
*   Freeze tf-estimator-nightly and keras-nightly for QA 5.6 branch[1def9d11a1e9ff9ffed9a73ceb46bbf62358185c](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/1def9d11a1e9ff9ffed9a73ceb46bbf62358185c)
*   Changes to track call-context information[5997aca821e964184a5cade3487fe50fe6b701ba](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/5997aca821e964184a5cade3487fe50fe6b701ba)
*   Fix unused and ignored status errors[7c3dcc03556fb5e2ddbc0a54454107d5544fa748](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/7c3dcc03556fb5e2ddbc0a54454107d5544fa748)
*   Avoid runtime errors with enhanced call context when mlir bridge is enabled[02c0323973d19b36bb0a9962df044e175cf83fc4](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/02c0323973d19b36bb0a9962df044e175cf83fc4)
*   Adjust mlir bridge check[01cec4dc473abea725915d2357457c60011de303](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/01cec4dc473abea725915d2357457c60011de303)
*   Only send call context for GPUs[ede927a9d3a99e6e1f2b80c802734781c7ee652a](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/ede927a9d3a99e6e1f2b80c802734781c7ee652a)
*   Fixes for call context feature[6d3b603caaa5184a73ee17598de9f16edd15eed6](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/6d3b603caaa5184a73ee17598de9f16edd15eed6)
*   [ROCm] Include rocm_config.h in rocm_dnn[c45f7a553f41398afb07572f501417fa64b5da16](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/c45f7a553f41398afb07572f501417fa64b5da16)
*   [ROCm]: Cleanup[78e276daaf8a833dfecfb6bc77c57dafad4592b6](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/78e276daaf8a833dfecfb6bc77c57dafad4592b6)
*   Revert "Migrate xla_test macro to use xla_cc_test instead of tf_cc_test"[63719b45142c3086745863ab0a6768c10687be9c](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/63719b45142c3086745863ab0a6768c10687be9c)
*   [ROCm]: Replace xla_cc_test/binary with tf_cc_test/binary[1e17b25377995d241665012f5707265a83f0882c](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/1e17b25377995d241665012f5707265a83f0882c)
*   fixed related matmul tests due to undefined attribute grad_a[99f7c94fbf64166f33d5c18bacafe22f291b9b82](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/99f7c94fbf64166f33d5c18bacafe22f291b9b82)
*   Adjust name for miopenkernels for 5.5[c88a9f4c5cd00ee8e30411d2309d3c56c62f1703](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/c88a9f4c5cd00ee8e30411d2309d3c56c62f1703)
*   Adding clang/17.0.0 to list of include dirs[3a1fc3b73a509b5ebb2a01f3f31cc89bc79b7ee5](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/3a1fc3b73a509b5ebb2a01f3f31cc89bc79b7ee5)
*   [ROCM] Fix includes for ROCm 5.6[f942c423b2210b9b07abd340a4d4b340e939fa01](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/f942c423b2210b9b07abd340a4d4b340e939fa01)
*   Adjust keras-nightly pin to consume RNN performance fix[e8cfb08777e8fad23fc6662b6ddfdea5ad52a07d](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/e8cfb08777e8fad23fc6662b6ddfdea5ad52a07d)
*   Fix python3 set up in Dockerfile.rocm[0436badc598c55f1d6391edfe1d5b3eb27132332](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/0436badc598c55f1d6391edfe1d5b3eb27132332)
*   More Replace xla_cc_test/binary with tf_cc_test/binary (cpu tests)[592c997a81da48a182bd3697d8c8d16fff33aaa6](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/592c997a81da48a182bd3697d8c8d16fff33aaa6)
*   Fix attr_builder_test[533a1e9eaa9cfe4a9a90e77e062ad671ab0f7ef1](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/commit/533a1e9eaa9cfe4a9a90e77e062ad671ab0f7ef1)
