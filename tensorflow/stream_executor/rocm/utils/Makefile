
HIPCC = /opt/rocm/bin/hipcc

HIPCC_FLAGS = -I /opt
HIPCC_FLAGS += -I ../../../../
#HIPCC_FLAGS += -g -ggdb -O0

rocm_collect_value_distribution_tester: rocm_collect_value_distribution_tester.cc rocm_collect_value_distribution.cc
	$(HIPCC) $(HIPCC_FLAGS) $^ -o $@
	./$@

clean:
	rm -rf rocm_collect_value_distribution_tester
