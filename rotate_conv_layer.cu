#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void RotateConvolutionLayer<Dtype>::Convolution_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top, const Dtype* weight) {	 
	for (int i = 0; i < bottom.size(); ++i) {
		const Dtype* bottom_data = bottom[i]->gpu_data();
		Dtype* top_data = top[i]->mutable_gpu_data();

		for (int n = 0; n < this->num_; ++n) {
			this->forward_gpu_gemm(bottom_data + bottom[i]->offset(n), weight,
			  top_data + top[i]->offset(n));

			if (this->bias_term_) {
				const Dtype* bias = this->blobs_[1]->gpu_data();
				this->forward_gpu_bias(top_data + top[i]->offset(n), bias);
			}
		}		
	}	
}

template <typename Dtype>
void RotateConvolutionLayer<Dtype>::MaxResponse_gpu(const vector<Blob<Dtype>*>& top) {
  int idx_max = -1;
  double max_response_norm = .0;

  // return the maximum response of rotational filters
  for (int r = 0; r < num_rotate_; ++r) {  
	double response_norm = .0;
	
	for (int i = 0; i < top.size(); ++i) {
		Dtype tmp_sum = 0.0;
		caffe_gpu_powx<Dtype>(top_rotate_[r][i]->count(), top_rotate_[r][i]->gpu_data(), 2.0, top[i]->mutable_gpu_data());
		caffe_gpu_asum<Dtype>(top[i]->count(), top[i]->gpu_data(), &tmp_sum);
		response_norm += tmp_sum;
	}
	
	if (response_norm > max_response_norm) {
		idx_max = r;
		max_response_norm = response_norm;
	}	
  }
  
  for (int i = 0; i < top.size(); ++i) {
	top[i]->CopyFrom(*(top_rotate_[idx_max][i]));
  }
}

template <typename Dtype>
void RotateConvolutionLayer<Dtype>::MeanResponse_gpu(const vector<Blob<Dtype>*>& top) {
  // return the mean response of rotational filters  
  for (int i = 0; i < top.size(); ++i) { 
	caffe_gpu_scal(top[i]->count(), static_cast<Dtype>(0.0), top[i]->mutable_gpu_data());
  
	for (int r = 0; r < num_rotate_; ++r) {
		caffe_gpu_add(top[i]->count(), top_rotate_[r][i]->gpu_data(), top[i]->gpu_data(), top[i]->mutable_gpu_data());
	}
	
	caffe_gpu_scal(top[i]->count(), static_cast<Dtype>(1.0/num_rotate_), top[i]->mutable_gpu_data());
  }
}

template <typename Dtype>
void RotateConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{ 
  // copy filter weights into OpenCV mat
  BlobToMat(*(this->blobs_[0]), weight_cv_mat_); 

  // forward pass with rotational filters
  if (rotate_mode_ == ConvolutionParameter_RotateMode_MAX_NORM) {  
	for (int r = 0; r < num_rotate_; ++r) {
		// rotate filters
		RotateFilters(rotate_angle_[r]);
		Convolution_gpu(bottom, top_rotate_[r], weight_warp_blob_.gpu_data());			
	}

	// post-processing response
	MaxResponse_gpu(top);
  }
  else if (rotate_mode_ == ConvolutionParameter_RotateMode_RAND) {			
	if (this->phase_ == TRAIN) {
		// randomly pick up a direction
		int idx = caffe_rng_rand() % num_rotate_;
		
		// rotate filters
		RotateFilters(rotate_angle_[idx]);
		Convolution_gpu(bottom, top_rotate_[idx], weight_warp_blob_.gpu_data());		
	}
	else if (this->phase_ == TEST) {
		for (int r = 0; r < num_rotate_; ++r) {
			// rotate filters
			RotateFilters(rotate_angle_[r]);
			Convolution_gpu(bottom, top_rotate_[r], weight_warp_blob_.gpu_data());			
		}
		
		MeanResponse_gpu(top);
	}		
	else {
		LOG(FATAL) << "Unknown phase.";
	}	
  }
  else {
	LOG(FATAL) << "Unknown rotation mode.";
  }
}

template <typename Dtype>
void RotateConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  if (this->param_propagate_down_[0]) {
    caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    caffe_gpu_set(this->blobs_[1]->count(), Dtype(0),
        this->blobs_[1]->mutable_gpu_diff());
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + top[i]->offset(n));
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + bottom[i]->offset(n),
              top_diff + top[i]->offset(n), weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + top[i]->offset(n), weight,
              bottom_diff + bottom[i]->offset(n));
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(RotateConvolutionLayer);

}  // namespace caffe
