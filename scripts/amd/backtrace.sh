sudo apt install gdb -y
pip3 install absl-py

options="$options --model=resnet50_v1.5"
options="$options --xla_compile"
options="$options --use_fp16"
options="$options --batch_size=256"
options="$options --print_training_accuracy"
options="$options --num_batches=1"
options="$options --num_gpus=8"
options="$options --variable_update=replicated"
options="$options --all_reduce_spec=nccl"

cd /dockerx/benchmarks

gdb -ex "set pagination off" \
    -ex "file python3" \
    -ex "run scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py ${options}" \
    -ex "backtrace" \
    -ex "set confirm off" \
    -ex "q" \
    2>&1 | tee /dockerx/pytorch/test_core_gdb.log