// rife implemented with ncnn library

#ifndef RIFE_OPS_H
#define RIFE_OPS_H

#include <vector>

// ncnn
#include "layer.h"
#include "pipeline.h"

class Warp : public ncnn::Layer
{
public:
    Warp();
    virtual int create_pipeline(const ncnn::Option& opt);
    virtual int destroy_pipeline(const ncnn::Option& opt);
    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const;
    virtual int forward(const std::vector<ncnn::VkMat>& bottom_blobs, std::vector<ncnn::VkMat>& top_blobs, ncnn::VkCompute& cmd, const ncnn::Option& opt) const;

private:
    ncnn::Pipeline* pipeline_warp;
    ncnn::Pipeline* pipeline_warp_pack4;
    ncnn::Pipeline* pipeline_warp_pack8;
};

class SplitFeature : public ncnn::Layer
{
public:
    SplitFeature();
    virtual int load_param(const ncnn::ParamDict& pd);
    virtual int create_pipeline(const ncnn::Option& opt);
    virtual int destroy_pipeline(const ncnn::Option& opt);
    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const;
    virtual int forward(const ncnn::VkMat& bottom_blob, ncnn::VkMat& top_blob, ncnn::VkCompute& cmd, const ncnn::Option& opt) const;
public:
    int stride;

private:
    ncnn::Pipeline* pipeline;

};

class MergeSplits: public ncnn::Layer
{
public:
    MergeSplits();
    virtual int load_param(const ncnn::ParamDict& pd);
    virtual int create_pipeline(const ncnn::Option& opt);
    virtual int destroy_pipeline(const ncnn::Option& opt);
    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const;
    virtual int forward(const ncnn::VkMat& bottom_blob, ncnn::VkMat& top_blob, ncnn::VkCompute& cmd, const ncnn::Option& opt) const;
public:
    int upscale_factor;
private:
    ncnn::Pipeline* pipeline;
};

class ConvexUpsampling : public ncnn::Layer
{
public:
    ConvexUpsampling();

    virtual int load_param(const ncnn::ParamDict& pd);

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const;

public:
    int split_num;
};

class Softsplat : public ncnn::Layer
{
public:
    Softsplat();

    virtual int load_param(const ncnn::ParamDict& pd);
    virtual int create_pipeline(const ncnn::Option& opt);
    virtual int destroy_pipeline(const ncnn::Option& opt);
    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const;
    virtual int forward(const std::vector<ncnn::VkMat>& bottom_blobs, std::vector<ncnn::VkMat>& top_blobs, ncnn::VkCompute& cmd, const ncnn::Option& opt) const;

private:
    ncnn::Pipeline* pipeline_init; // fill zero
    ncnn::Pipeline* pipeline_softsplat;
    ncnn::Pipeline* pipeline_softsplat_pack4;
    ncnn::Pipeline* pipeline_softsplat_pack8;
    
    ncnn::Pipeline* pipeline_norm;
    ncnn::Pipeline* pipeline_norm_pack4;
    ncnn::Pipeline* pipeline_norm_pack8;

};

class Gt : public ncnn::Layer
{
public:
    Gt();

    virtual int load_param(const ncnn::ParamDict& pd);

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const;

};

#endif // RIFE_OPS_H
