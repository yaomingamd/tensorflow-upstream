#pragma once

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class LaunchRsqrtEpsOp
{
public:
      void operator()(const Device& d,
          typename TTypes<T>::Tensor out,
          typename TTypes<T>::ConstTensor in,
          T eps) {
         out.device(d) = in.constant(T(1)) / (in.sqrt() + in.constant(eps));
      }
};

template <typename Device, typename T>
class LaunchRsqrtEpsGradOp
{
public:
      void operator()(const Device& d,
          typename TTypes<T>::Tensor out,
          typename TTypes<T>::ConstTensor in,
          typename TTypes<T>::ConstTensor grad,
          T eps) {

      	// y = 1/(sqrt(x)+c)
      	// y' = -0.5/(sqrt(x)+c)^2 sqrt(x)
      	auto sq = in.sqrt();
      	out.device(d) = grad * in.constant(T(-0.5)) / (sq * (sq+in.constant(eps)).square());
      }
};

template <typename T>
class LaunchRsqrtEpsOp<GPUDevice, T>
{
public:
      void operator()(const GPUDevice& d,
          typename TTypes<T>::Tensor out,
          typename TTypes<T>::ConstTensor in,
          T eps);
};

template <typename T>
class LaunchRsqrtEpsGradOp<GPUDevice, T>
{
public:
      void operator()(const GPUDevice& d,
          typename TTypes<T>::Tensor out,
          typename TTypes<T>::ConstTensor in,
          typename TTypes<T>::ConstTensor grad,
          T eps);
};


};