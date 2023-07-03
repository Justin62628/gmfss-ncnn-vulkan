// rife implemented with ncnn library

#include "rife_ops.h"
#include "split_feature.comp.hex.h"

using namespace ncnn;

SplitFeature::SplitFeature()
{
    support_vulkan = false;
    one_blob_only = true;
    support_inplace = false;
    pipeline = 0;
}

int SplitFeature::load_param(const ParamDict& pd)
{
    stride = pd.get(0, 2);

    return 0;
}

int SplitFeature::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    int outw = w / stride; // w=w h=h d=1 c=c -> w=w/s h=h/s d=c c=s*s
    int outh = h / stride;
    int outd = channels;
    int outc = stride * stride;

    top_blob.create(outw, outh, outd, outc, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)  // top.d
    {
        const Mat m = bottom_blob.channel(q);

        for (int sh = 0; sh < stride; sh++)
        {
            for (int sw = 0; sw < stride; sw++)
            {
                int p = sh * stride + sw;

                for (int i = 0; i < outh; i++)
                {
                    const float* sptr = m.row(i + sh * outh ) + sw*outw;
                    float* outptr = top_blob.channel(p).depth(q).row(i); // at c=p, d=q (w,h) plane

                    //float* outptr = top_blob.row((p * channels + q) * outw * outh + i * outh); // at c=p, d=q (w,h) plane
                    for (int j = 0; j < outw; j++)
                    {
                        outptr[0] = sptr[0];

                        sptr ++;
                        outptr++;
                    }
                }
            }
        }
    }

    return 0;
}

int SplitFeature::forward(const ncnn::VkMat& bottom_blob, ncnn::VkMat& top_blob, ncnn::VkCompute& cmd, const ncnn::Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int c = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    VkMat input;
    input.create(w, h, c * elempack, elemsize / elempack, 1, opt.blob_vkallocator);
    if (input.empty())
        return -100;
    vkdev->convert_packing(bottom_blob, input, 1, cmd, opt);
    c = input.c;
    elempack = input.elempack;  // should be 1
    elemsize = input.elemsize;
    
    int outw = w / stride; // w=w h=h d=1 c=c -> w=w/s h=h/s d=c c=s*s
    int outh = h / stride;
    int outd = c;
    int outc = stride * stride;

    top_blob.create(outw, outh, outd, outc, elemsize, elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;
    
    {
        std::vector<VkMat> bindings(2);
        bindings[0] = input;
        bindings[1] = top_blob;

        std::vector<vk_constant_type> constants(8);
        constants[0].i = input.w;
        constants[1].i = input.h;
        constants[2].i = input.c;
        constants[3].i = stride;
        constants[4].i = top_blob.w;
        constants[5].i = top_blob.h;
        constants[6].i = top_blob.d;
        constants[7].i = top_blob.c;

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);
    }

    return 0;
    
}

int SplitFeature::create_pipeline(const Option& opt)
{
    if (!vkdev)
        return 0;

    std::vector<vk_specialization_type> specializations(0 + 0);

    {
        // pack1
        static std::vector<uint32_t> spirv;
        static ncnn::Mutex lock;
        {
            ncnn::MutexLockGuard guard(lock);
            if (spirv.empty())
            {
                compile_spirv_module(split_feature_comp_data, sizeof(split_feature_comp_data), opt, spirv);
            }
        }

        pipeline = new Pipeline(vkdev);
        pipeline->set_optimal_local_size_xyz();
        pipeline->create(spirv.data(), spirv.size() * 4, specializations);
    }

    return 0;
}

int SplitFeature::destroy_pipeline(const Option& opt)
{
    // pack1
    delete pipeline;
    pipeline = 0;
    return 0;
}