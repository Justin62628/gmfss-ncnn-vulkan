// rife implemented with ncnn library

#include "rife_ops.h"
#include "merge_splits.comp.hex.h"

using namespace ncnn;

MergeSplits::MergeSplits()
{
    support_vulkan = false;
    one_blob_only = true;
    support_inplace = false;
    pipeline = 0;
}

int MergeSplits::load_param(const ParamDict& pd)
{
    upscale_factor = pd.get(0, 2);

    return 0;
}

int MergeSplits::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    int outw = w * upscale_factor;
    int outh = h * upscale_factor;
    int outc = d;

    top_blob.create(outw, outh, outc, elemsize, opt.blob_allocator);
    if (top_blob.empty() || channels != upscale_factor * upscale_factor)
        return -100;

#pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outc; p++)
    {
        Mat m = top_blob.channel(p);

        for (int sh = 0; sh < upscale_factor; sh++)
        {
            for (int sw = 0; sw < upscale_factor; sw++)
            {
                int q;
                q = sh * upscale_factor + sw;

                const Mat s = bottom_blob.channel(q);
                for (int i = 0; i < h; i++)
                {
                    float* outptr = m.row(i + sh * h) + sw * w;
                    const float* sptr = s.channel(p).row(i);
                    for (int j = 0; j < w; j++)
                    {
                        outptr[0] = sptr[0];

                        sptr++;
                        outptr++;
                    }
                }
            }
        }
    }

    return 0;
}

int MergeSplits::forward(const ncnn::VkMat& bottom_blob, ncnn::VkMat& top_blob, ncnn::VkCompute& cmd, const ncnn::Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    VkMat input;
    input.create(w, h, d, channels * elempack, elemsize / elempack, 1, opt.blob_vkallocator);
    if (input.empty())
        return -100;
    vkdev->convert_packing(bottom_blob, input, 1, cmd, opt);
    channels = input.c;
    elempack = input.elempack;  // should be 1
    elemsize = input.elemsize;

    int outw = w * upscale_factor;
    int outh = h * upscale_factor;
    int outc = d;

    top_blob.create(outw, outh, outc, elemsize, opt.blob_vkallocator);
    if (top_blob.empty() || channels != upscale_factor * upscale_factor)
        return -100;
    
    {
        std::vector<VkMat> bindings(2);
        bindings[0] = input;
        bindings[1] = top_blob;

        std::vector<vk_constant_type> constants(8);
        constants[0].i = input.w;
        constants[1].i = input.h;
        constants[2].i = input.d;
        constants[3].i = input.c;
        constants[4].i = upscale_factor;
        constants[5].i = top_blob.w;
        constants[6].i = top_blob.h;
        constants[7].i = top_blob.c;

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);
    }

    return 0;
    
}

int MergeSplits::create_pipeline(const Option& opt)
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
                compile_spirv_module(merge_splits_comp_data, sizeof(merge_splits_comp_data), opt, spirv);
            }
        }

        pipeline = new Pipeline(vkdev);
        pipeline->set_optimal_local_size_xyz();
        pipeline->create(spirv.data(), spirv.size() * 4, specializations);
    }

    return 0;
}

int MergeSplits::destroy_pipeline(const Option& opt)
{
    // pack1
    delete pipeline;
    pipeline = 0;
    return 0;
}