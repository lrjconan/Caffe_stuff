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
void RotateConvolutionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {  
  // call LayerSetUp from base class
  BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  
  // initialize member variables   
  forward_count_ = 0;
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();  
  num_rotate_ 	= conv_param.num_rotate();
  min_angle_ 	= conv_param.min_angle();
  max_angle_ 	= conv_param.max_angle();
  rotate_mode_ 	= conv_param.rotate_mode();
  rotate_gap_ 	= conv_param.rotate_gap();
  
  CHECK_GT(num_rotate_, 0) << "Filter dimensions should be greater than zero.";
  CHECK_GE(max_angle_, min_angle_) << "Max angle should be no less than min angle.";
  
  rotate_angle_.resize(num_rotate_);  
  rotate_angle_[0] = min_angle_;
  
  if (num_rotate_ > 1) {
	Dtype angle_gap = (max_angle_ - min_angle_) / static_cast<Dtype>(num_rotate_ - 1);
	for (int i = 1; i < num_rotate_; ++i) {
		rotate_angle_[i] = rotate_angle_[i-1] + angle_gap;
	}
  }
  
  const int filter_num 		= this->blobs_[0]->num();
  const int filter_channel 	= this->blobs_[0]->channels();
  const int filter_height 	= this->blobs_[0]->height();
  const int filter_width 	= this->blobs_[0]->width();
  
  weight_warp_blob_.Reshape(filter_num, filter_channel, filter_height, filter_width);
  weight_cv_mat_.resize(filter_num);
  weight_warp_cv_mat_.resize(filter_num);
 
  for (int i = 0; i < weight_cv_mat_.size(); ++i) {
	weight_cv_mat_[i].resize(filter_channel);
	weight_warp_cv_mat_[i].resize(filter_channel);
	
	for (int j = 0; j < weight_cv_mat_[i].size(); ++j) {
		if (sizeof(Dtype) == 4) {
			weight_cv_mat_[i][j] 		= cv::Mat(filter_height, filter_width, CV_32FC1);
			weight_warp_cv_mat_[i][j] 	= cv::Mat(filter_height, filter_width, CV_32FC1);
		}
		else if (sizeof(Dtype) == 8) {
			weight_cv_mat_[i][j] 		= cv::Mat(filter_height, filter_width, CV_64FC1);
			weight_warp_cv_mat_[i][j] 	= cv::Mat(filter_height, filter_width, CV_64FC1);			
		}
		else {
			LOG(FATAL) << "Unknown data type.";
		}
	}
  }
}

template <typename Dtype>
void RotateConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // call Reshape from base class
  BaseConvolutionLayer<Dtype>::Reshape(bottom, top);

  // initialize result buffer for rotational convolution  
  if (top_rotate_.size() == 0) {
	top_rotate_.resize(num_rotate_);
	for (int i = 0; i < num_rotate_; ++i) {	
		top_rotate_[i].resize(top.size());

		for (int j = 0; j < top.size(); ++j) {
			top_rotate_[i][j] = new Blob<Dtype>(top[j]->num(), top[j]->channels(), top[j]->height(), top[j]->width());
		}
	}  
  }
}

template <typename Dtype>
void RotateConvolutionLayer<Dtype>::compute_output_shape() {
  this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->kernel_h_)
      / this->stride_h_ + 1;
  this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->kernel_w_)
      / this->stride_w_ + 1;
}

template <typename Dtype>
void RotateConvolutionLayer<Dtype>::Rotate(const cv::Mat& src, const Dtype angle, cv::Mat& dst) {
	int len = std::max(src.cols, src.rows);
	cv::Point2f pt(len/2., len/2.);
	cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);
	cv::warpAffine(src, dst, r, cv::Size(len, len));
}

template <typename Dtype>
void RotateConvolutionLayer<Dtype>::RotateFilters(const Dtype angle) {
	for (int i = 0; i < weight_cv_mat_.size(); ++i) {
		for (int j = 0; j < weight_cv_mat_[i].size(); ++j) 
		{	
			Rotate(weight_cv_mat_[i][j], angle, weight_warp_cv_mat_[i][j]);

			/*
			// save filter weights
			cv::Mat write_mat(weight_warp_cv_mat_[i][j]);
			double minVal, maxVal;
			minMaxLoc(write_mat, &minVal, &maxVal); //find minimum and maximum intensities
			write_mat.convertTo(write_mat, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));			
			char str_name[100];
			sprintf(str_name, "./weight/conv_weight_n=%03d_c=%03d_angle=%4.1f.png", i, j, angle);
			cv::imwrite(str_name, write_mat);
			*/
		}
	}
	
	// copy to a blob
	MatToBlob(weight_warp_cv_mat_, weight_warp_blob_);
}

template <typename Dtype>
void RotateConvolutionLayer<Dtype>::Convolution_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top, const Dtype* weight) {	 
	for (int i = 0; i < bottom.size(); ++i) {
		const Dtype* bottom_data = bottom[i]->cpu_data();
		Dtype* top_data = top[i]->mutable_cpu_data();

		for (int n = 0; n < this->num_; ++n) {
			this->forward_cpu_gemm(bottom_data + bottom[i]->offset(n), weight,
			  top_data + top[i]->offset(n));

			if (this->bias_term_) {
				const Dtype* bias = this->blobs_[1]->cpu_data();
				this->forward_cpu_bias(top_data + top[i]->offset(n), bias);
			}
		}		
	}	
}

template <typename Dtype>
void RotateConvolutionLayer<Dtype>::MaxResponse_cpu(const vector<Blob<Dtype>*>& top) {
  int idx_max 			= -1;
  double max_response_norm 	= .0;

  // return the maximum response of rotational filters
  for (int r = 0; r < num_rotate_; ++r) {  
	double response_norm = .0;
	
	for (int i = 0; i < top.size(); ++i) {  
		caffe_powx<Dtype>(top_rotate_[r][i]->count(), top_rotate_[r][i]->cpu_data(), 2.0, top[i]->mutable_cpu_data());
		response_norm += caffe_cpu_asum<Dtype>(top[i]->count(), top[i]->cpu_data());
	}
	
	if (response_norm > max_response_norm) {
		idx_max 		= r;
		max_response_norm 	= response_norm;
	}	
  }
  
  for (int i = 0; i < top.size(); ++i) {
	top[i]->CopyFrom(*(top_rotate_[idx_max][i]));
  }
}

template <typename Dtype>
void RotateConvolutionLayer<Dtype>::MeanResponse_cpu(const vector<Blob<Dtype>*>& top) {
  // return the mean response of rotational filters  
  for (int i = 0; i < top.size(); ++i) { 
	caffe_scal(top[i]->count(), static_cast<Dtype>(0.0), top[i]->mutable_cpu_data());
  
	for (int r = 0; r < num_rotate_; ++r) {
		caffe_add(top[i]->count(), top_rotate_[r][i]->cpu_data(), top[i]->cpu_data(), top[i]->mutable_cpu_data());
	}
	
	caffe_scal(top[i]->count(), static_cast<Dtype>(1.0/num_rotate_), top[i]->mutable_cpu_data());
  }
}

template <typename Dtype>
void RotateConvolutionLayer<Dtype>::BlobToMat(const Blob<Dtype>& src, std::vector<std::vector<cv::Mat> >& dst) 
{
  // Mat is assumed to be the type of CV_32FC1
  const int blob_num 		= src.num();
  const int blob_channel 	= src.channels();
  const int blob_height 	= src.height();
  const int blob_width 		= src.width();

  const int mat_num 		= dst.size();
  const int mat_channel 	= (dst[0]).size();
  const int mat_height 		= (dst[0][0]).rows;
  const int mat_width 		= (dst[0][0]).cols;
  
  CHECK_EQ(blob_num, mat_num) << "Blob and mat should be the same size!";
  CHECK_EQ(blob_channel, mat_channel) << "Blob and mat should be the same size!";
  CHECK_EQ(blob_height, mat_height) << "Blob and mat should be the same size!";
  CHECK_EQ(blob_width, mat_width) << "Blob and mat should be the same size!";
  
  const Dtype* blob_ptr = src.cpu_data();
    
  for (int n = 0; n < blob_num; ++n) {	
	for (int c = 0; c < blob_channel; ++c) {	
		Dtype* mat_ptr = (dst[n][c]).ptr<Dtype>(0);		
		caffe_copy(blob_height*blob_width, blob_ptr, mat_ptr);
		blob_ptr += blob_height*blob_width;
		
		/*
		for (int h = 0; h < blob_height; ++h) {		
			float* row_ptr = (dst[n][c]).ptr<float>(h);
		
			for (int w = 0; w < blob_width; ++w)
				row_ptr[w] = blob_ptr[w];
			
			blob_ptr += blob_width;
		}
		*/
	}
  }    
}

template <typename Dtype>
void RotateConvolutionLayer<Dtype>::MatToBlob(const std::vector<std::vector<cv::Mat> >& src, Blob<Dtype>& dst)
{
  // Mat is assumed to be the type of CV_32FC1
  const int blob_num 		= dst.num();
  const int blob_channel 	= dst.channels();
  const int blob_height 	= dst.height();
  const int blob_width 		= dst.width();

  const int mat_num 		= src.size();
  const int mat_channel 	= (src[0]).size();
  const int mat_height 		= (src[0][0]).rows;
  const int mat_width 		= (src[0][0]).cols;
  
  CHECK_EQ(blob_num, mat_num) << "Blob and mat should be the same size!";
  CHECK_EQ(blob_channel, mat_channel) << "Blob and mat should be the same size!";
  CHECK_EQ(blob_height, mat_height) << "Blob and mat should be the same size!";
  CHECK_EQ(blob_width, mat_width) << "Blob and mat should be the same size!";
  
  Dtype* blob_ptr = dst.mutable_cpu_data();
  
  for (int n = 0; n < blob_num; ++n) {	
	for (int c = 0; c < blob_channel; ++c) {
		const Dtype* mat_ptr = (src[n][c]).ptr<Dtype>(0);
		caffe_copy(blob_height*blob_width, mat_ptr, blob_ptr);
		blob_ptr += blob_height*blob_width;
		
		/*
		for (int h = 0; h < blob_height; ++h) {		
			const float* row_ptr = (src[n][c]).ptr<float>(h);
			
			for (int w = 0; w < blob_width; ++w)
				blob_ptr[w] = row_ptr[w];
			
			blob_ptr += blob_width;
		}
		*/
	}
  }
}

template <typename Dtype>
void RotateConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{ 
  // copy filter weights into OpenCV mat
  BlobToMat(*(this->blobs_[0]), weight_cv_mat_); 

  // forward pass with rotational filters
  if (rotate_mode_ == ConvolutionParameter_RotateMode_MAX_NORM) {  
	for (int r = 0; r < num_rotate_; ++r) {
		// rotate filters
		RotateFilters(rotate_angle_[r]);
		Convolution_cpu(bottom, top_rotate_[r], weight_warp_blob_.cpu_data());			
	}

	// post-processing response
	MaxResponse_cpu(top);
  }
  else if (rotate_mode_ == ConvolutionParameter_RotateMode_RAND) {			
	if (this->phase_ == TRAIN) {
		// randomly pick up a direction
		int idx = caffe_rng_rand() % num_rotate_;
		
		// rotate filters
		RotateFilters(rotate_angle_[idx]);
		Convolution_cpu(bottom, top_rotate_[idx], weight_warp_blob_.cpu_data());		
	}
	else if (this->phase_ == TEST) {
		for (int r = 0; r < num_rotate_; ++r) {
			// rotate filters
			RotateFilters(rotate_angle_[r]);
			Convolution_cpu(bottom, top_rotate_[r], weight_warp_blob_.cpu_data());			
		}
		
		MeanResponse_cpu(top);
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
void RotateConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  if (this->param_propagate_down_[0]) {
    caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    caffe_set(this->blobs_[1]->count(), Dtype(0),
        this->blobs_[1]->mutable_cpu_diff());
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + top[i]->offset(n));
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + bottom[i]->offset(n),
              top_diff + top[i]->offset(n), weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + top[i]->offset(n), weight,
              bottom_diff + bottom[i]->offset(n));
        }
      }
    }
  }
}

template <typename Dtype>
RotateConvolutionLayer<Dtype>::~RotateConvolutionLayer() {
  for (int i = 0; i < num_rotate_; ++i) {  
	for (int j = 0; j < top_rotate_[i].size(); ++j) {		
		delete top_rotate_[i][j];
	}
  }
}

#ifdef CPU_ONLY
STUB_GPU(RotateConvolutionLayer);
#endif

INSTANTIATE_CLASS(RotateConvolutionLayer);
REGISTER_LAYER_CLASS(RotateConvolution);

}  // namespace caffe
