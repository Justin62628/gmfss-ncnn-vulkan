// rife implemented with ncnn library

#include "rife_ops.h"

#include "softsplat_init.comp.hex.h"
#include "softsplat.comp.hex.h"
#include "softsplat_norm.comp.hex.h"

using namespace ncnn;

Softsplat::Softsplat()
{
    support_vulkan = true;
    support_inplace = false;
    support_packing = false;
    pipeline_init = 0;
    pipeline_softsplat = 0;
    pipeline_norm = 0;
    pipeline_softsplat_pack4 = 0; // TODO
    pipeline_softsplat_pack8 = 0;
}


int Softsplat::load_param(const ParamDict& pd)
{
    return 0;
}

int Softsplat::forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
{
    const Mat& input = bottom_blobs[0];
    const Mat& flow = bottom_blobs[1];

    int w = input.w;
    int h = input.h;
    int c = input.c;
    size_t elemsize = input.elemsize;

    Mat& top_blob = top_blobs[0];
    top_blob.create(w, h, c-1, elemsize, opt.blob_allocator);
    // top_blob.fill(0);
    ncnn::Mat ss_blob;
    ss_blob.create(w, h, c, elemsize, opt.blob_allocator);
    ss_blob.fill(0);
    if (top_blob.empty())
        return -100;

#pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < c; q++)
    {
        Mat m = ss_blob.channel(q);
        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < w; j++)
            {
                float fltOutputX = j + flow.channel(0).row(i)[j];
                float fltOutputY = i + flow.channel(1).row(i)[j];

                int intNorthwestX = (int)(floor(fltOutputX));
                int intNorthwestY = (int)(floor(fltOutputY));
                int intNortheastX = intNorthwestX + 1;
                int intNortheastY = intNorthwestY;
                int intSouthwestX = intNorthwestX;
                int intSouthwestY = intNorthwestY + 1;
                int intSoutheastX = intNorthwestX + 1;
                int intSoutheastY = intNorthwestY + 1;

                float fltNorthwest = ((float)(intSoutheastX)-fltOutputX) * ((float)(intSoutheastY)-fltOutputY);
                float fltNortheast = (fltOutputX - (float)(intSouthwestX)) * ((float)(intSouthwestY)-fltOutputY);
                float fltSouthwest = ((float)(intNortheastX)-fltOutputX) * (fltOutputY - (float)(intNortheastY));
                float fltSoutheast = (fltOutputX - (float)(intNorthwestX)) * (fltOutputY - (float)(intNorthwestY));

                if (intNorthwestX >= 0 && intNorthwestX < w && intNorthwestY >= 0 && intNorthwestY < h)
                    m.row(intNorthwestY)[intNorthwestX] += input.channel(q).row(i)[j] * fltNorthwest;
                if (intNortheastX >= 0 && intNortheastX < w && intNortheastY >= 0 && intNortheastY < h)
                    m.row(intNortheastY)[intNortheastX] += input.channel(q).row(i)[j]*fltNortheast;
                if (intSouthwestX >= 0 && intSouthwestX < w && intSouthwestY >= 0 && intSouthwestY < h)
                    m.row(intSouthwestY)[intSouthwestX] += input.channel(q).row(i)[j] * fltSouthwest;
                if (intSoutheastX >= 0 && intSoutheastX < w && intSoutheastY >= 0 && intSoutheastY < h)
                    m.row(intSoutheastY)[intSoutheastX] += input.channel(q).row(i)[j] * fltSoutheast;
            }
        }
    }
    // top_blob = ss_blob.channel_range(0, c-1) / (ss_blob.channel(-1) + 0.00001);
#pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < c - 1; q++)
    {
        Mat m = top_blob.channel(q);
        Mat n = ss_blob.channel(q);
        Mat d = ss_blob.channel(c - 1);  // last channel
        for (int i = 0; i < h; i++)
        {
            float* ptr = m.row(i);
            const float* ptr1 = n.row(i);
            const float* ptr2 = d.row(i);
            for (int j = 0; j < w; j++)
            {
                ptr[j] = ptr1[j] / (ptr2[j] + 0.00001);
            }
        }
    }

    return 0;
}

int Softsplat::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkMat& input_unpacked = bottom_blobs[0];
    const VkMat& flow = bottom_blobs[1];

    int w = input_unpacked.w;
    int h = input_unpacked.h;
    int c = input_unpacked.c;
    size_t elemsize = input_unpacked.elemsize;
    int elempack = input_unpacked.elempack;

    VkMat input;
    input.create(w, h, c * elempack, elemsize / elempack, 1, opt.blob_vkallocator);
    if (input.empty())
        return -100;
    vkdev->convert_packing(input_unpacked, input, 1, cmd, opt);
    c = input.c;
    elempack = input.elempack;  // should be 1
    elemsize = input.elemsize;

    VkMat& top_blob = top_blobs[0];
    top_blob.create(w, h, c-1, elemsize, elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    VkMat ss_blob;
    // ss_blob.create(w, h, c, elemsize, elempack, opt.blob_vkallocator);
    ss_blob.create(w, h, c, 4u, elempack, opt.blob_vkallocator);  // float
    if (ss_blob.empty())
        return -100;

    {
        // init, fill ss_blob with zero
        std::vector<ncnn::VkMat> bindings(1);
        bindings[0] = ss_blob;

        std::vector<ncnn::vk_constant_type> constants(5);
        constants[0].i = ss_blob.w;
        constants[1].i = ss_blob.h;
        constants[2].i = ss_blob.c;
        constants[3].i = ss_blob.cstep;
        constants[4].f = 0.f;

        cmd.record_pipeline(pipeline_init, bindings, constants, ss_blob);
    }

    {
        // softsplat
        std::vector<VkMat> bindings(3);
        bindings[0] = input;
        bindings[1] = flow;
        bindings[2] = ss_blob;

        std::vector<vk_constant_type> constants(4);
        constants[0].i = ss_blob.w;
        constants[1].i = ss_blob.h;
        constants[2].i = ss_blob.c;
        constants[3].i = ss_blob.cstep;

        cmd.record_pipeline(pipeline_softsplat, bindings, constants, ss_blob);
    }

    {
        // norm
        std::vector<VkMat> bindings(2);
        bindings[0] = ss_blob;
        bindings[1] = top_blob;

        std::vector<vk_constant_type> constants(4);
        constants[0].i = top_blob.w;
        constants[1].i = top_blob.h;
        constants[2].i = top_blob.c;
        constants[3].i = top_blob.cstep;

        cmd.record_pipeline(pipeline_norm, bindings, constants, top_blob);
    }

    return 0;
}


int Softsplat::create_pipeline(const Option& opt)
{
    if (!vkdev)
        return 0;

    std::vector<vk_specialization_type> specializations(0 + 0);

    // init
    {
        // pack1
        static std::vector<uint32_t> spirv;
        static ncnn::Mutex lock;
        {
            ncnn::MutexLockGuard guard(lock);
            if (spirv.empty())
            {
                compile_spirv_module(softsplat_init_comp_data, sizeof(softsplat_init_comp_data), opt, spirv);
            }
        }

        pipeline_init = new Pipeline(vkdev);
        pipeline_init->set_optimal_local_size_xyz();
        pipeline_init->create(spirv.data(), spirv.size() * 4, specializations);
    }

    // softsplat
    {
        // pack1
        static std::vector<uint32_t> spirv;
        static ncnn::Mutex lock;
        {
            ncnn::MutexLockGuard guard(lock);
            if (spirv.empty())
            {
                compile_spirv_module(softsplat_comp_data, sizeof(softsplat_comp_data), opt, spirv);
            }
        }

        pipeline_softsplat = new Pipeline(vkdev);
        pipeline_softsplat->set_optimal_local_size_xyz();
        pipeline_softsplat->create(spirv.data(), spirv.size() * 4, specializations);
    }

    // norm
    {
        // pack1
        static std::vector<uint32_t> spirv;
        static ncnn::Mutex lock;
        {
            ncnn::MutexLockGuard guard(lock);
            if (spirv.empty())
            {
                compile_spirv_module(softsplat_norm_comp_data, sizeof(softsplat_norm_comp_data), opt, spirv);
            }
        }

        pipeline_norm = new Pipeline(vkdev);
        pipeline_norm->set_optimal_local_size_xyz();
        pipeline_norm->create(spirv.data(), spirv.size() * 4, specializations);
    }

    return 0;
}

int Softsplat::destroy_pipeline(const Option& opt)
{
    // pack1
    delete pipeline_init;
    pipeline_init = 0;
    delete pipeline_softsplat;
    pipeline_softsplat = 0;
    delete pipeline_norm;
    pipeline_norm = 0;

    delete pipeline_softsplat_pack4;
    pipeline_softsplat_pack4 = 0;

    delete pipeline_softsplat_pack8;
    pipeline_softsplat_pack8 = 0;

    return 0;
}