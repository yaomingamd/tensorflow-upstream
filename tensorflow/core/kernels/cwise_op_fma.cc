/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "tensorflow/core/kernels/cwise_op_fma.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T, int SGN>
void LaunchFusedMulAddOp<Device,T,SGN>::operator()(const Device& device,
		T* out,
		const T* x1, const T* y1, const T* x2,
		uint64 elements,
		bool broadcast_x1, bool broadcast_y1, 
		bool broadcast_x2)
{
	for(uint64 i=0; i<elements; i++)
	{
    T m1 = x1[broadcast_x1 ? 0 : i]*y1[broadcast_y1 ? 0 : i];
    T m2 = x2[broadcast_x2 ? 0 : i];
    if(SGN>0)
		  out[i] = m1+m2;
    else if(SGN<0)
      out[i] = m1-m2;
    else
      out[i] = m2-m1;
	}
}

template <typename Device, typename T, int SGN>
void LaunchFusedMulAdd2Op<Device,T,SGN>::operator()(const Device& device,
	T* out,
	const T* x1, const T* y1, const T* x2, const T* y2,
	uint64 elements,
	bool broadcast_x1, bool broadcast_y1, 
	bool broadcast_x2, bool broadcast_y2)
{
	for(uint64 i=0; i<elements; i++)
	{
    T m1 = x1[broadcast_x1 ? 0 : i]*y1[broadcast_y1 ? 0 : i];
    T m2 = x2[broadcast_x2 ? 0 : i]*y2[broadcast_y2 ? 0 : i];
		out[i] = (SGN>0) ? m1+m2 : m1-m2;
	}
}

template <typename Device, typename T, int SGN>
void FallbackLaunchFusedMulAddOp<Device, T, SGN>::operator()(const Device& device,
    T* out,
    const T* x1, const T* y1, const T* x2,
    int64 dims[5],
    uint8 broadcast_masks[5])
{
  //printf("FallbackLaunchFusedMulAddOpCPU\n");
  int64 strides[3][4];
  for(int i=0; i<3; i++)
  {
    int64 s = 1;
    for(int j=0; j<4; j++) {
      int b = broadcast_masks[j] & (1<<i);
      int b_next = broadcast_masks[j+1] & (1<<i);
      s *= b ? dims[j] : 1;
      strides[i][j] = s * ((b_next - b)>>i);
    }
  };
  for(uint64 u=0; u<dims[4]; u++) {
    for(uint64 v=0; v<dims[3]; v++) {
      for(uint64 z=0; z<dims[2]; z++) {
        for(uint64 y=0; y<dims[1]; y++) {
          for(uint64 x=0; x<dims[0]; x++) {
            *out = (SGN>0)
              ? (*x1)*(*y1)+(*x2)
              : ((SGN<0) 
                 ? (*x1)*(*y1)-(*x2)
                 : (*x2)-(*x1)*(*y1));
            out++;
            if(broadcast_masks[0] & 1)
              x1++;
            if(broadcast_masks[0] & 2)
              y1++;
            if(broadcast_masks[0] & 4)
              x2++;
          }
          x1 += strides[0][0]; 
          y1 += strides[1][0];
          x2 += strides[2][0];
        }
        x1 += strides[0][1]; 
        y1 += strides[1][1];
        x2 += strides[2][1];
      }
      x1 += strides[0][2]; 
      y1 += strides[1][2];
      x2 += strides[2][2];
    }
    x1 += strides[0][3]; 
    y1 += strides[1][3];
    x2 += strides[2][3];
  }
}


template <typename Device, typename T, int SGN>
void FallbackLaunchFusedMulAdd2Op<Device, T, SGN>::operator()(const Device& device,
    T* out,
    const T* x1, const T* y1, const T* x2, const T* y2,
    int64 dims[5],
    uint8 broadcast_masks[5])
{
  int64 strides[4][4];
  for(int i=0; i<4; i++)
  {
    int64 s = 1;
    for(int j=0; j<4; j++) {
      int b = broadcast_masks[j] & (1<<i);
      int b_next = broadcast_masks[j+1] & (1<<i);
      s *= b ? dims[j] : 1;
      strides[i][j] = s * ((b_next - b)>>i);
    }
  };
  for(uint64 u=0; u<dims[4]; u++) {
    for(uint64 v=0; v<dims[3]; v++) {
      for(uint64 z=0; z<dims[2]; z++) {
        for(uint64 y=0; y<dims[1]; y++) {
          for(uint64 x=0; x<dims[0]; x++) {
            *out = (SGN>0)
              ? (*x1)*(*y1)+(*x2)*(*y2)
              : (*x1)*(*y1)-(*x2)*(*y2);
            out++;
            if(broadcast_masks[0] & 1)
              x1++;
            if(broadcast_masks[0] & 2)
              y1++;
            if(broadcast_masks[0] & 4)
              x2++;
            if(broadcast_masks[0] & 8)
              y2++;
          }
          x1 += strides[0][0]; 
          y1 += strides[1][0];
          x2 += strides[2][0];
          y2 += strides[3][0];
        }
        x1 += strides[0][1]; 
        y1 += strides[1][1];
        x2 += strides[2][1];
        y2 += strides[3][1];
      }
      x1 += strides[0][2]; 
      y1 += strides[1][2];
      x2 += strides[2][2];
      y2 += strides[3][2];
    }
    x1 += strides[0][3]; 
    y1 += strides[1][3];
    x2 += strides[2][3];
    y2 += strides[3][3];
  }
}

template <typename Device, int N>
class FusedMulAddBase {
 public:
  // Analyze the incoming shapes for compatibility, calculate
  // the output shape and the necessary broadcasts.
  bool DoShapeAnalysis(OpKernelContext* ctx, 
    const Tensor** inputs,
    bool& pure_broadcast,
    uint8* broadcast_masks,
    int64* out_dims,
    TensorShape& out_shape,
    int64& out_elements) {
    int64 in_elements[N];
    int in_ranks[N];
    int rank=0;
 
    for(int i=0; i<N; i++) {
      in_elements[i] = inputs[i]->NumElements();
      in_ranks[i] = inputs[i]->dims();
      rank = (rank>in_ranks[i]) ? rank : in_ranks[i];
 /*     
      printf("Input %d: %d dimensions, %d elements\n", i, in_ranks[i], in_elements[i]);
      for(int j=0; j<in_ranks[i]; j++)
        printf("%d ", inputs[i]->shape().dim_size(j));
      printf("\n");
*/      
    }
    if(rank>5)
      return false;
    //for(int i=0; i<N; i++) 
    //  if(in_ranks[i]!=0 && in_ranks[i]!=1 && in_ranks[i]!=dims)
    //    return false;
/*
    OP_REQUIRES(ctx, dims<=5, errors::InvalidArgument("FusedMulAddOp does "
      "not support ", dims, "dimensions"));
    for(int i=0; i<N; i++) 
      OP_REQUIRES(ctx, (in_ranks[i]==1 || in_ranks[i]==dims),
        errors::InvalidArgument("FusedMulAddOp with inputs of incompatible "
        "dimension counts ", in_ranks[i], dims));
*/        


    int64 xds[5][N];
    for(int i=0; i<rank; i++) {
      int64 max_dim = 0;
      int ii = rank-i-1;
      for(int j=0; j<N; j++) {
        if(ii < in_ranks[j])
          xds[i][j] = inputs[j]->shape().dim_size(in_ranks[j]-ii-1);
        else
          xds[i][j] = 1;
        if(xds[i][j] == 0)
          xds[i][j] = 1;
        max_dim = (max_dim>xds[i][j]) ? max_dim : xds[i][j];
      }
      for(int j=0; j<N; j++) {
        if(!(xds[i][j]==1 || xds[i][j]==max_dim))
          return false;
        broadcast_masks[rank-i-1]|=(xds[i][j]!=1 ? 1 : 0)<<j;
      }
      max_dim = (max_dim>1) ? max_dim : 1;
      out_shape.AddDim(max_dim);
      out_dims[rank-i-1]=max_dim;
    }
    out_elements = out_shape.num_elements();
    pure_broadcast = true;
    for(int i=0; i<N; i++)
      if(in_elements[i]!=1 && in_elements[i]!=out_elements)
        pure_broadcast = false;
    if(pure_broadcast) {
      broadcast_masks[0]=0;
      for(int i=0; i<N; i++)
        if(in_elements[i]!=1)
          broadcast_masks[0]|=1<<i;
      return true;
    }
    /*
    printf("%d elements, broadcast %s\n", out_elements, pure_broadcast?"true":"false");
    for(int i=0; i<5; i++)      
      printf("out_dim[%d] = %d mask %d\n", i, out_dims[i], (int)broadcast_masks[i]);
    */
    // folding dimensions from the highest down
    // [50,10,10]x[50,10,10]x[50,1,1] -> [50,100]x[50,100]x[50,1]
    bool folded = false;
    for(int i=rank-1; i>0; i--) {
      if(out_dims[i]==1)
        continue;
      if(broadcast_masks[i] == broadcast_masks[i-1]) {
        folded = true;
        out_dims[i-1]*=out_dims[i];
        for(int j=i; j<rank-1; j++) {
          out_dims[j] = out_dims[j+1];
          broadcast_masks[j] = broadcast_masks[j+1];
        }
        out_dims[rank-1]=1;
        broadcast_masks[rank-1]=0;
      }
    }
    /*
    if(folded) {
      printf("After folding:\n");
      for(int i=0; i<5; i++)
        printf("out_dim[%d] = %d mask %d\n", i, out_dims[i], (int)broadcast_masks[i]);
    }
    */
    return true;

  }
};


template <typename Device, typename T, int SGN=1>
class FusedMulAddOp : public OpKernel, public FusedMulAddBase<Device, 3> {
 public:
  explicit FusedMulAddOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor* inputs[3];
    //printf("FusedMulAddOp\n");
    OP_REQUIRES_OK(ctx, ctx->input("x1", &inputs[0]));
    OP_REQUIRES_OK(ctx, ctx->input("y1", &inputs[1]));
    OP_REQUIRES_OK(ctx, ctx->input("x2", &inputs[2]));
    bool pure_broadcast=true;
    uint8 broadcast_masks[5]={0,0,0,0,0};
    int64 out_dims[5]={1,1,1,1,1};
    TensorShape out_shape;
    int64 out_elements=0;
    bool ok = FusedMulAddBase<Device,3>::DoShapeAnalysis(ctx, inputs, pure_broadcast, broadcast_masks, out_dims,
      out_shape, out_elements);
    OP_REQUIRES(ctx, ok, errors::InvalidArgument("FusedMulAdd with incompatible shapes"));
    //pure_broadcast=false;
    Tensor* output = nullptr;
    // todo: an OP_REQUIRES to check that all dims fit in 32 bit
  	OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));
    if(pure_broadcast)
    	LaunchFusedMulAddOp<Device,T,SGN>()(ctx->eigen_device<Device>(), 
    		output->flat<T>().data(), 
    		inputs[0]->flat<T>().data(), inputs[1]->flat<T>().data(), 
    		inputs[2]->flat<T>().data(), 
    		out_elements, 
    		!(broadcast_masks[0]&1), !(broadcast_masks[0]&2),
    		!(broadcast_masks[0]&4));
    else
      FallbackLaunchFusedMulAddOp<Device,T,SGN>()(ctx->eigen_device<Device>(), 
        output->flat<T>().data(), 
        inputs[0]->flat<T>().data(), inputs[1]->flat<T>().data(), 
        inputs[2]->flat<T>().data(), 
        out_dims, broadcast_masks);
  }
};

template <typename Device, typename T, int SGN=1>
class FusedMulAdd2Op : public OpKernel, public FusedMulAddBase<Device, 4> {
 public:
  explicit FusedMulAdd2Op(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor* inputs[4];
    //printf("FusedMulAdd2\n");
    OP_REQUIRES_OK(ctx, ctx->input("x1", &inputs[0]));
    OP_REQUIRES_OK(ctx, ctx->input("y1", &inputs[1]));
    OP_REQUIRES_OK(ctx, ctx->input("x2", &inputs[2]));
    OP_REQUIRES_OK(ctx, ctx->input("y2", &inputs[3]));
    bool pure_broadcast=true;
    uint8 broadcast_masks[5]={0,0,0,0,0};
    int64 out_dims[5]={1,1,1,1,1};
    TensorShape out_shape;
    int64 out_elements=0;
    bool ok = FusedMulAddBase<Device,4>::DoShapeAnalysis(ctx, inputs, pure_broadcast, broadcast_masks, out_dims,
      out_shape, out_elements);
    OP_REQUIRES(ctx, ok, errors::InvalidArgument("FusedMulAdd2 with incompatible shapes"));
    //pure_broadcast=false;
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));
    if(pure_broadcast)
      LaunchFusedMulAdd2Op<Device,T,SGN>()(ctx->eigen_device<Device>(), 
        output->flat<T>().data(), 
        inputs[0]->flat<T>().data(), inputs[1]->flat<T>().data(), 
        inputs[2]->flat<T>().data(), inputs[3]->flat<T>().data(),
        out_elements, 
        !(broadcast_masks[0]&1), !(broadcast_masks[0]&2),
        !(broadcast_masks[0]&4), !(broadcast_masks[0]&8));
    else
      FallbackLaunchFusedMulAdd2Op<Device,T,SGN>()(ctx->eigen_device<Device>(), 
        output->flat<T>().data(), 
        inputs[0]->flat<T>().data(), inputs[1]->flat<T>().data(), 
        inputs[2]->flat<T>().data(), inputs[3]->flat<T>().data(),
        out_dims, broadcast_masks);
  }
};

#define REGISTER_CPU_KERNEL(type)                                          \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("FusedMulAdd").Device(DEVICE_CPU).TypeConstraint<type>("T"),    \
      FusedMulAddOp<CPUDevice, type>);                                     \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("FusedMulAdd2").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
      FusedMulAdd2Op<CPUDevice, type>);                                    \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("FusedMulSub").Device(DEVICE_CPU).TypeConstraint<type>("T"),    \
      FusedMulAddOp<CPUDevice, type, -1>);                                 \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("FusedMulSubRev").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      FusedMulAddOp<CPUDevice, type, 0>);                                  \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("FusedMulSub2").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
      FusedMulAdd2Op<CPUDevice, type, -1>);

REGISTER_CPU_KERNEL(Eigen::half);
REGISTER_CPU_KERNEL(float);
REGISTER_CPU_KERNEL(double);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNEL(type)                                          \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("FusedMulAdd").Device(DEVICE_GPU).TypeConstraint<type>("T"),    \
      FusedMulAddOp<GPUDevice, type>);                                     \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("FusedMulAdd2").Device(DEVICE_GPU).TypeConstraint<type>("T"),   \
      FusedMulAdd2Op<GPUDevice, type>);                                    \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("FusedMulSub").Device(DEVICE_GPU).TypeConstraint<type>("T"),    \
      FusedMulAddOp<GPUDevice, type, -1>);                                 \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("FusedMulSubRev").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      FusedMulAddOp<GPUDevice, type, 0>);                                  \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("FusedMulSub2").Device(DEVICE_GPU).TypeConstraint<type>("T"),   \
      FusedMulAdd2Op<GPUDevice, type, -1>);

REGISTER_GPU_KERNEL(Eigen::half);
REGISTER_GPU_KERNEL(float);
REGISTER_GPU_KERNEL(double);
#endif
};