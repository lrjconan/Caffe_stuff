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
void RotateConvolutionLayer<Dtype>::compute_output_shape() {
  this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->kernel_h_)
      / this->stride_h_ + 1;
  this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->kernel_w_)
      / this->stride_w_ + 1;
}

template <typename Dtype>
void RotateConvolutionLayer<Dtype>::Rotate(const cv::Mat& src, const Dtype angle, cv::Mat& dst) 
{
    int len = std::max(src.cols, src.rows);
    cv::Point2f pt(len/2., len/2.);
    cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);
    cv::warpAffine(src, dst, r, cv::Size(len, len));
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
		for (int h = 0; h < blob_height; ++h) {		
			float* row_ptr = (dst[n][c]).ptr<float>(h);
		
			for (int w = 0; w < blob_width; ++w)
				row_ptr[w] = blob_ptr[w];
			
			blob_ptr += blob_width;
		}
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
		for (int h = 0; h < blob_height; ++h) {		
			const float* row_ptr = (src[n][c]).ptr<float>(h);
		
			for (int w = 0; w < blob_width; ++w)
				blob_ptr[w] = row_ptr[w];
			
			blob_ptr += blob_width;
		}
	}
  }
}

template <typename Dtype>
void RotateConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{ 
  // copy filters into OpenCV mats
  const int filter_num 		= this->blobs_[0]->num();
  const int filter_channel 	= this->blobs_[0]->channels();
  const int filter_height 	= this->blobs_[0]->height();
  const int filter_width 	= this->blobs_[0]->width();
   
  std::vector<std::vector<cv::Mat> > weight_mat, weight_mat_copy;   
  weight_mat.resize(filter_num);
  weight_mat_copy.resize(filter_num);
 
  for (int i = 0; i < weight_mat.size(); ++i) {
	weight_mat[i].resize(filter_channel);
	weight_mat_copy[i].resize(filter_channel);
	
	for (int j = 0; j < weight_mat[i].size(); ++j) {
		weight_mat[i][j] = cv::Mat(filter_height, filter_width, CV_32FC1);
		weight_mat_copy[i][j] = cv::Mat(filter_height, filter_width, CV_32FC1);
	}
  }
	
  BlobToMat(*(this->blobs_[0]), weight_mat); 
  Blob<Dtype> weight_blob_copy(filter_num, filter_channel, filter_height, filter_width);
 
  vector<Blob<Dtype>*> top_copy(top);
 
  // forward pass
  double max_response_norm = .0;
  const double const_denominator = static_cast<Dtype>(bottom.size()*this->num_);
  
  for (int r = 0; r < BaseConvolutionLayer<Dtype>::num_rotate_; ++r) {
    // rotate filters
	if (r > 0) {
		for (int i = 0; i < weight_mat.size(); ++i) {
			for (int j = 0; j < weight_mat[i].size(); ++j) 
			{	
				Rotate(weight_mat[i][j], BaseConvolutionLayer<Dtype>::rotate_angle_[r], weight_mat_copy[i][j]);
				
				/*
				cv::Mat write_mat(weight_mat_copy[i][j]);
				write_mat.convertTo(write_mat, CV_8UC1, 255.0);
				char str_name[100];
				sprintf(str_name, "./conv_weight_n=%03d_c=%03d_r=%d.png", i, j, r);
				cv::imwrite(str_name, write_mat);					
				*/
			}
		}			
	}

	MatToBlob(weight_mat_copy, weight_blob_copy);
	const Dtype* weight = weight_blob_copy.cpu_data();
		
	double response_norm = .0;
	for (int i = 0; i < bottom.size(); ++i) 
	{
		const Dtype* bottom_data = bottom[i]->cpu_data();
		Dtype* top_data = top_copy[i]->mutable_cpu_data();

		for (int n = 0; n < this->num_; ++n) 
		{
		  this->forward_cpu_gemm(bottom_data + bottom[i]->offset(n), weight,
			  top_data + top_copy[i]->offset(n));
		  
		  if (this->bias_term_) 
		  {
			const Dtype* bias = this->blobs_[1]->cpu_data();
			this->forward_cpu_bias(top_data + top_copy[i]->offset(n), bias);
		  }
		}
		
		caffe_powx<Dtype>(top_copy[i]->count(), top_copy[i]->cpu_data(), 2.0, top_copy[i]->mutable_cpu_data());
		caffe_scal<Dtype>(top_copy[i]->count(), 1.0/const_denominator, top_copy[i]->mutable_cpu_data());
		
		response_norm += caffe_cpu_asum<Dtype>(top_copy[i]->count(), top_copy[i]->cpu_data());		
	}

	if (response_norm > max_response_norm) {
		for (int i = 0; i < bottom.size(); ++i) {
			top[i]->CopyFrom(*(top_copy[i]));
		}
	}
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

#ifdef CPU_ONLY
STUB_GPU(RotateConvolutionLayer);
#endif

INSTANTIATE_CLASS(RotateConvolutionLayer);
REGISTER_LAYER_CLASS(RotateConvolution);

}  // namespace caffe
