#include <hip/hip_runtime.h>
#include <limits>
namespace stream_executor {
namespace gpu {

template <typename T, bool propagate_nans>
__device__ bool compare(T acc, T val)
{
	return propagate_nans ? !(val <= acc) : val > acc;
}

template <typename T, int Mode, bool propagate_nans>
__device__ void reduce(T& acc, T val)
{
	if(Mode)
		acc += val;
	else if (propagate_nans ? !(val <= acc) : val > acc)
		acc = val;
}

template <typename T, int Mode,  bool propagate_nans>
__global__ void poolingYKernel(T* out, const T* in,
	int batches, int width, int height, 
			int window_x, int window_y,
			int width_out, int height_out,
			int padding_x, int stride_x, 
			int padding_y, int stride_y)
{
	int bid = blockIdx.x * blockDim.z + threadIdx.z;
	if(bid>=batches)
		return;
	
	in += bid * width * height;
	out += bid * width_out * height_out;

	 for(int y=threadIdx.y; y<height_out; y+=blockDim.y)
	 	for(int x=threadIdx.x; x<width_out; x+=blockDim.x)
	 	{	 		
	 		int x0 = x;
	 		int y0 = y*stride_y-padding_y;
	 		const T* p = in + x0 + y0*width;
	 		T val = Mode ? 0 : std::numeric_limits<T>::lowest();
	 		int min_dy = max(0, -y0);
	 		int max_dy = min(window_y, height-y0);

	 		for(int dy=min_dy; dy<max_dy; dy++)
	 			reduce<T,Mode,propagate_nans>(val, p[dy*width]);
			if(Mode)
			{
				int pool_count = max_dy-min_dy;
				if(pool_count==0)
					pool_count=1;
				val *= T(1.0)/pool_count;
			}
			out[x+y*width_out] = val;
	 	}
}

// Pooling performed as two 1-D reductions with results passed in shared memory.
// Significantly outperforms the basic version for large windows. Assumes stride_y=1.
template <typename T, int Mode, bool propagate_nans>
__global__ void pooling2DKernelSHM(T* out, const T* in,
	int batches, int width, int height, 
			int window_x, int window_y,
			int width_out, int height_out,
			int padding_x, int stride_x, 
			int padding_y, int stride_y)
{
	int bid = blockIdx.x * blockDim.z + threadIdx.z;
	in += bid * width * height;
	out += bid * width_out * height_out;

	__shared__ T part_reduce[1024];
	int y_out_group = blockDim.y-window_y;
	for(int y_=0; y_<height; y_+=y_out_group)
	{
		int y = y_ + threadIdx.y;
		int sid = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
		for(int x_ = 0; x_ < width_out; x_ += blockDim.x)
		{
			int x = x_ + threadIdx.x;
			int x0 = x*stride_x-padding_x;
			int y0 = y*stride_y-padding_y;
			const T* p = in + x0 + y0*width;
			int min_dx = max(0, -x0);
			int min_dy = max(0, -y0);
			int max_dx = min(window_x, width-x0);
			int max_dy = min(window_y, height-y0);
			p += min_dy*width;
			if(bid<batches && x<width_out && y<height)
			{
				T self = Mode ? 0 : std::numeric_limits<T>::lowest();
				for(int dx=min_dx; dx<max_dx; dx++)
					reduce<T,Mode,propagate_nans>(self, p[dx]);
				part_reduce[sid] = self;
			}
			__syncthreads();

			if(bid<batches && x<width_out && y<height_out && threadIdx.y<y_out_group)
			{	
				T self = part_reduce[sid];			
				for(int dy=1; dy<max_dy-min_dy; dy++)
					reduce<T,Mode,propagate_nans>(self, part_reduce[sid+dy*blockDim.x]);
				if(Mode)
				{
					int pool_count = (max_dx-min_dx)*(max_dy-min_dy);
					if(pool_count==0)
						pool_count=1;
					self *= T(1.0)/pool_count;
				}
				out[x+y*width_out] = self;
			}
			__syncthreads();
	 	}
	 }
}

template <typename T, int Mode, bool propagate_nans>
__global__ void pooling2DKernel(T* out, const T* in,
	int batches, int width, int height, 
			int window_x, int window_y,
			int width_out, int height_out,
			int padding_x, int stride_x, 
			int padding_y, int stride_y)
{
	int bid = blockIdx.x * blockDim.z + threadIdx.z;
	if(bid>=batches)
		return;

	in += bid * width * height;
	out += bid * width_out * height_out;

	 for(int y=threadIdx.y; y<height_out; y+=blockDim.y)
	 	for(int x=threadIdx.x; x<width_out; x+=blockDim.x)
	 	{	 		
	 		int x0 = x*stride_x-padding_x;
	 		int y0 = y*stride_y-padding_y;
	 		const T* p = in + x0 + y0*width;
	 		int min_dx = max(0, -x0);
	 		int min_dy = max(0, -y0);
	 		int max_dx = min(window_x, width-x0);
	 		int max_dy = min(window_y, height-y0);
	 		p += min_dy*width;
	 		//T val = Mode ? 0 : *p;
	 		T val = Mode ? 0 : std::numeric_limits<T>::lowest();
	 		for(int dy=min_dy; dy<max_dy; dy++)
	 		{
	 			const T* q = p;
	 			for(int dx=min_dx; dx<max_dx; dx++)
	 				reduce<T,Mode,propagate_nans>(val,q[dx]);
				p+=width;
			}
			if(Mode)
			{
				int pool_count = (max_dx-min_dx)*(max_dy-min_dy);
				if(pool_count==0)
					pool_count=1;
				val *= T(1.0)/pool_count;
			}
			out[x+y*width_out] = val;
	 	}
}

template <typename T, int Mode, bool propagate_nans>
__global__ void pooling2DBackpropKernel(const T* out, const T* in,
	const T* pgrad_out, T* pgrad_in,
	int batches, int width, int height, 
			int window_x, int window_y,
			int width_out, int height_out,
			int padding_x, int stride_x, 
			int padding_y, int stride_y)
{
	int bid = blockIdx.x * blockDim.z + threadIdx.z;
	if(bid>=batches)
		return;

	in += bid * width * height;
	out += bid * width_out * height_out;
	pgrad_in += bid * width * height;
	pgrad_out += bid * width_out * height_out;

	 for(int y=threadIdx.y; y<height_out; y+=blockDim.y)
	 	for(int x=threadIdx.x; x<width_out; x+=blockDim.x)
	 	{	 		
	 		int x0 = x*stride_x-padding_x;
	 		int y0 = y*stride_y-padding_y;
	 		const T* p = in + x0 + y0*width;
	 		T* pgin = pgrad_in + x0 + y0*width;
	 		int min_dx = max(0, -x0);
	 		int min_dy = max(0, -y0);
	 		int max_dx = min(window_x, width-x0);
	 		int max_dy = min(window_y, height-y0);

	 		T outgrad = pgrad_out[x + y*width_out];
	 		T outval = out[x+y*width_out];

	 		T scaler = 0.0;
	 		if(Mode)
	 		{
	 			int pool_count = (max_dx-min_dx)*(max_dy-min_dy);
	 			if(pool_count==0)
	 				pool_count=1;
	 			scaler = T(1.0)/pool_count;
	 		}

	 		// in case of a tie in maxpooling, we propagate to the first input in raster order
	 		// (otherwise TF tests will fail)
	 		T maxval; 
	 		int index;
	 		if(propagate_nans) {
	 			index = min_dx + min_dy*width;
	 			maxval = p[index];
	 		}
	 		else {
	 			index = -1;
	 			maxval = std::numeric_limits<T>::lowest();
	 		}
	 		for(int dy=min_dy; dy<max_dy; dy++)
	 			for(int dx=min_dx; dx<max_dx; dx++)
				{
					T v = p[dx+dy*width];
					if(Mode)
						atomicAdd(&pgin[dx+dy*width], scaler * outgrad);
					else 
					{
						if(compare<T,propagate_nans>(maxval, v) && !isnan(v))
						{
							maxval = v;
							index = dx+dy*width;
						}
					}
				}
			if(Mode==0 && index!=-1)
				atomicAdd(&pgin[index], outgrad);
	 	}
}

template <class T, int Mode, bool propagate_nans>
void rocmPooling2DImpl(void* stream, T* out, const T* in,
	int batches, int width, int height, 
			int window_x, int window_y,
			int width_out, int height_out,
			int padding_x, int stride_x,
			int padding_y, int stride_y)
{
	int grid_x = min(1024, width_out);
	int grid_y = min(1024/grid_x, height_out);

	// 1-D kernel
	if(window_x>=4 && window_y>=4 && height-1+window_y<1024 && stride_y==1) {
		grid_y = height-1+window_y;
		grid_x = min(1024/grid_y, width_out);
	}

	int grid_z = 1024 / (grid_x * grid_y);
	int blocks = (batches+grid_z-1) / grid_z;
    hipLaunchKernelGGL(
    	(window_x==1 && padding_x==0 && stride_x==1) 
    			? poolingYKernel<T,Mode,propagate_nans>
    		: ((window_x>=4 && window_y>=4 && height-1+window_y<1024 && stride_y==1)
    			? pooling2DKernelSHM<T,Mode,propagate_nans>
    			: pooling2DKernel<T,Mode,propagate_nans>),
    	blocks, dim3(grid_x, grid_y, grid_z), 0,
        (hipStream_t)stream,
        out, in, batches, width, height, window_x, window_y, width_out, height_out, 
        padding_x, stride_x, padding_y, stride_y);
}

template <class T>
void rocmPooling2D(void* stream, T* out, const T* in,
	int batches, int width, int height, 
			int window_x, int window_y,
			int width_out, int height_out,
			int padding_x, int stride_x,
			int padding_y, int stride_y, bool maxpool, bool propagate_nans)
{
	auto impl = maxpool
		? (propagate_nans ? rocmPooling2DImpl<T,0,true> :rocmPooling2DImpl<T,0,false>)
		: (propagate_nans ? rocmPooling2DImpl<T,1,true> :rocmPooling2DImpl<T,1,false>);
	impl(stream, out, in, batches, width, height,
		window_x, window_y, width_out, height_out,
		padding_x, stride_x, padding_y, stride_y);
} 

template <class T>
void rocmBackwardPooling2D(void* stream, const T* out, const T* in,
			const T* pgrad_out, T* pgrad_in,
			int batches, int width, int height, 	
			int window_x, int window_y,
			int width_out, int height_out,
			int padding_x, int stride_x,
			int padding_y, int stride_y, bool maxpool, bool propagate_nans)
{
	auto kernel = maxpool 
		? (propagate_nans ? pooling2DBackpropKernel<T,0,true> : pooling2DBackpropKernel<T,0,false> )
		: (propagate_nans ? pooling2DBackpropKernel<T,1,true> : pooling2DBackpropKernel<T,1,false> );
	int grid_x, grid_y;
	hipMemsetAsync(pgrad_in, 0, batches*width*height*sizeof(T), (hipStream_t)stream);
	grid_x = min(32, width_out);
	grid_y = min(32, height_out);
	int grid_z = 1024 / (grid_x * grid_y);
	int blocks = (batches+grid_z-1) / grid_z;
    hipLaunchKernelGGL(
        kernel, blocks, dim3(grid_x, grid_y, grid_z), 0,
        (hipStream_t)stream,
        out, in, pgrad_out, pgrad_in, batches, width, height, window_x, window_y, width_out, height_out, 
        padding_x, stride_x, padding_y, stride_y);
} 


template
void rocmPooling2D<float>(void* stream, float* out, const float* in,
	int batches, int width, int height, 
			int window_x, int window_y,
			int width_out, int height_out,
			int padding_x, int stride_x,
			int padding_y, int stride_y, bool maxpool, bool propagate_nans);
template 
void rocmPooling2D<double>(void* stream, double* out, const double* in,
	int batches, int width, int height, 
			int window_x, int window_y,
			int width_out, int height_out,
			int padding_x, int stride_x,
			int padding_y, int stride_y, bool maxpool, bool propagate_nans);

template 
void rocmBackwardPooling2D<float>(void* stream, const float* out, const float* in,
			const float* pgrad_out, float* pgrad_in,
			int batches, int width, int height, 
			int window_x, int window_y,
			int width_out, int height_out,
			int padding_x, int stride_x,
			int padding_y, int stride_y, bool maxpool, bool propagate_nans);

template 
void rocmBackwardPooling2D<double>(void* stream, const double* out, const double* in,
			const double* pgrad_out, double* pgrad_in,
			int batches, int width, int height, 
			int window_x, int window_y,
			int width_out, int height_out,
			int padding_x, int stride_x,
			int padding_y, int stride_y, bool maxpool, bool propagate_nans);

};
};
