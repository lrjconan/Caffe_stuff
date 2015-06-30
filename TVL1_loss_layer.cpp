#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
// TVL1 loss layer solved by Iterative Reweighted Least Square (IRLS) method
// added by Renjie

// gradient in x direction : [0 -1 1]
template <typename Dtype>
void TVL1LossLayer<Dtype>::GradientX(const Blob<Dtype> & imgs, Blob<Dtype> & gX) 
{	
  CHECK_EQ(imgs.num()     , gX.num()     );
  CHECK_EQ(imgs.channels(), gX.channels());
  CHECK_EQ(imgs.height()  , gX.height()  );
  CHECK_EQ(imgs.width()   , gX.width()   );
 
  int num(imgs.num()), channels(imgs.channels()), height(imgs.height()), width(imgs.width());

  const Dtype * imgRowPtr	= imgs.cpu_data();
  Dtype * gXRowPtr			= gX.mutable_cpu_data();
  const int rowBuffSize 	= width;

  for (int i = 0; i < num; ++i) {
	  for (int c = 0; c < channels; ++c) {		
		  for (int h = 0; h < height; ++h) {			  
			  for (int w = 0; w < width-1; ++w) 
				  gXRowPtr[w] = imgRowPtr[w+1] - imgRowPtr[w];
			  		
			  // circular shift
			  gXRowPtr[width-1] = imgRowPtr[0] - imgRowPtr[width-1];
				
			  imgRowPtr	+= rowBuffSize;
			  gXRowPtr	+= rowBuffSize;
		  }
	  }
  }
}

// gradient in y direction : [0 -1 1]^T
template <typename Dtype>
void TVL1LossLayer<Dtype>::GradientY(const Blob<Dtype> & imgs, Blob<Dtype> & gY) 
{	
  CHECK_EQ(imgs.num()     , gY.num()     );
  CHECK_EQ(imgs.channels(), gY.channels());
  CHECK_EQ(imgs.height()  , gY.height()  );
  CHECK_EQ(imgs.width()   , gY.width()   );
 
  int num(imgs.num()), channels(imgs.channels()), height(imgs.height()), width(imgs.width());

  const Dtype * imgPtr			= imgs.cpu_data();
  const Dtype * imgRowPtr		= imgPtr;
  const Dtype * imgRowNextPtr	= imgPtr;
  Dtype * gYRowPtr				= gY.mutable_cpu_data();
  
  const int rowBuffSize = width;
  const int imgBuffSize = height*width;
  imgRowNextPtr += rowBuffSize;

  for (int i = 0; i < num; ++i) {
	  for (int c = 0; c < channels; ++c) {
		  for (int h = 0; h < height-1; ++h) {
			  for (int w = 0; w < width; ++w) 				  
				  gYRowPtr[w] = imgRowNextPtr[w] - imgRowPtr[w];
			  
			  gYRowPtr		+= rowBuffSize;
			  imgRowPtr		+= rowBuffSize;
			  imgRowNextPtr += rowBuffSize;
		  }

		  // circular shift
		  for (int w = 0; w < width; ++w)				  
			gYRowPtr[w]		= imgPtr[w] - imgRowPtr[w];
			
		  gYRowPtr		+= rowBuffSize;
		  imgRowPtr		+= rowBuffSize;
		  imgRowNextPtr	+= rowBuffSize;
          imgPtr		+= imgBuffSize;
	  }
  }
}

template <typename Dtype>
void TVL1LossLayer<Dtype>::UpdateWeight(const Blob<Dtype> & diff_data, 
	const Blob<Dtype> & img_dx, const Blob<Dtype> & img_dy) 
{
  const int count			= diff_data.count();
  const Dtype* data_ptr		= diff_data.cpu_data(); 
  const Dtype* dx_ptr		= img_dx.cpu_data();  
  const Dtype* dy_ptr		= img_dy.cpu_data();  

  Dtype* weight_data_ptr	= weight_data_.mutable_cpu_data(); 
  Dtype* weight_dx_ptr		= weight_tv_x_.mutable_cpu_data();  
  Dtype* weight_dy_ptr		= weight_tv_y_.mutable_cpu_data();  

  // data format: ((n * channels_ + c) * height_ + h) * width_ + w
  for (int i = 0; i < count; ++i) {	  
	weight_data_ptr[i]	= 1.0/std::max(delta_, fabs(data_ptr[i]));
	weight_dx_ptr[i]	= 1.0/std::max(delta_, fabs(dx_ptr[i]));
	weight_dy_ptr[i]	= 1.0/std::max(delta_, fabs(dy_ptr[i]));
  }
}

template <typename Dtype>
void TVL1LossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  LossLayer<Dtype>::LayerSetUp(bottom, top);
     
  CHECK_EQ(bottom.size(), 2) << "Loss Layer takes two blobs as input.";
  CHECK_EQ(bottom[0]->num()     , bottom[1]->num()     )<< "The data and label should have the same number.";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height()  , bottom[1]->height()  );
  CHECK_EQ(bottom[0]->width()   , bottom[1]->width()   );

  forward_count_ 	= 0;
  is_update_		= false;
  lambda_			= this->layer_param_.tvl1_loss_param().lambda();
  delta_			= this->layer_param_.tvl1_loss_param().delta();
  tv_show_			= this->layer_param_.tvl1_loss_param().tv_show();
  update_iter_		= this->layer_param_.tvl1_loss_param().update_iter();

  diff_data_.Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  img_dx_.Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  img_dy_.Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());

  weight_data_.Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  weight_tv_x_.Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  weight_tv_y_.Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());

  caffe_set(weight_data_.count(), Dtype(1.0), weight_data_.mutable_cpu_data());
  caffe_set(weight_tv_x_.count(), Dtype(1.0), weight_tv_x_.mutable_cpu_data());
  caffe_set(weight_tv_y_.count(), Dtype(1.0), weight_tv_y_.mutable_cpu_data()); 
}

template <typename Dtype>
void TVL1LossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  LossLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void TVL1LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  int count	= bottom[0]->count();  // num_ * channels_ * height_ * width_;
  int num		= bottom[0]->num();

  if (forward_count_ % update_iter_ == 0) 
  {
	is_update_ 		= true;
	forward_count_ 	= 0;
  } 
  
  // calculate difference
  caffe_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), diff_data_.mutable_cpu_data());

  // calculate derivative of image
  GradientX(*(bottom[0]), img_dx_);
  GradientY(*(bottom[0]), img_dy_);
  
  Dtype tv_term =  caffe_cpu_asum(count, img_dx_.cpu_data());
  tv_term		+= caffe_cpu_asum(count, img_dy_.cpu_data());
  tv_term		*= Dtype(lambda_/num);
  
  if (tv_show_)	LOG(INFO) << "TV = " << tv_term;

  Dtype loss	=  caffe_cpu_asum(count, diff_data_.cpu_data());
  loss			/= Dtype(num);
  loss			+= tv_term;
     
  top[0]->mutable_cpu_data()[0] = loss;
  
  forward_count_++;
}

template <typename Dtype>
void TVL1LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  if (propagate_down[1]) 
  {
    LOG(FATAL) << this->type() << "Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) 
  {
	  int count	= bottom[0]->count();
	  int num	= bottom[0]->num();

	  if (is_update_) {
		UpdateWeight(diff_data_, img_dx_, img_dy_);
		is_update_ = false;
	  }
		
	  // gradient of data term
	  caffe_mul(diff_data_.count(), diff_data_.cpu_data(), weight_data_.cpu_data(), diff_data_.mutable_cpu_data());

	  // gradient of TV term
	  caffe_mul(img_dx_.count(), img_dx_.cpu_data(), weight_tv_x_.cpu_data(), img_dx_.mutable_cpu_data());
	  caffe_mul(img_dy_.count(), img_dy_.cpu_data(), weight_tv_y_.cpu_data(), img_dy_.mutable_cpu_data());

	  // Update the gradient  
	  caffe_cpu_axpby(count, Dtype(2.0) / num, diff_data_.cpu_data(), Dtype(0), bottom[0]->mutable_cpu_diff());  
	  caffe_cpu_axpby(count, Dtype(2.0*lambda_/num), img_dx_.cpu_data(), Dtype(1.0), bottom[0]->mutable_cpu_diff());
	  caffe_axpy(count, Dtype(2.0*lambda_/num), img_dy_.cpu_data(), bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(TVL1LossLayer);
#endif

INSTANTIATE_CLASS(TVL1LossLayer);
REGISTER_LAYER_CLASS(TVL1Loss);

}  // namespace caffe
