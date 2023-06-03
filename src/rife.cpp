// rife implemented with ncnn library

#include "rife.h"

#include <algorithm>
#include <vector>
#include "benchmark.h"

#include "rife_preproc.comp.hex.h"
#include "rife_postproc.comp.hex.h"
#include "rife_preproc_tta.comp.hex.h"
#include "rife_postproc_tta.comp.hex.h"
#include "rife_flow_tta_avg.comp.hex.h"
#include "rife_v2_flow_tta_avg.comp.hex.h"
#include "rife_v4_flow_tta_avg.comp.hex.h"
#include "rife_flow_tta_temporal_avg.comp.hex.h"
#include "rife_v2_flow_tta_temporal_avg.comp.hex.h"
#include "rife_v4_flow_tta_temporal_avg.comp.hex.h"
#include "rife_out_tta_temporal_avg.comp.hex.h"
#include "rife_v4_timestep.comp.hex.h"
#include "rife_v4_timestep_tta.comp.hex.h"

#include "rife_ops.h"
#include "layer.h"
#include "layer_type.h"

DEFINE_LAYER_CREATOR(Warp)
DEFINE_LAYER_CREATOR(SplitFeature)
DEFINE_LAYER_CREATOR(MergeSplits)
DEFINE_LAYER_CREATOR(ConvexUpsampling)


RIFE::RIFE(int gpuid, bool _tta_mode, bool _tta_temporal_mode, bool _uhd_mode, int _num_threads, bool _rife_v2, bool _rife_v4)
{
    vkdev = gpuid == -1 ? 0 : ncnn::get_gpu_device(gpuid);

    rife_preproc = 0;
    rife_postproc = 0;
    rife_flow_tta_avg = 0;
    rife_flow_tta_temporal_avg = 0;
    rife_out_tta_temporal_avg = 0;
    rife_v4_timestep = 0;
    rife_uhd_downscale_image = 0;
    rife_uhd_upscale_flow = 0;
    rife_uhd_double_flow = 0;
    rife_v2_slice_flow = 0;
    tta_mode = _tta_mode;
    tta_temporal_mode = _tta_temporal_mode;
    uhd_mode = _uhd_mode;
    num_threads = _num_threads;
    rife_v2 = _rife_v2;
    rife_v4 = _rife_v4;
}

RIFE::~RIFE()
{
    // cleanup preprocess and postprocess pipeline
    {
        delete rife_preproc;
        delete rife_postproc;
        delete rife_flow_tta_avg;
        delete rife_flow_tta_temporal_avg;
        delete rife_out_tta_temporal_avg;
        delete rife_v4_timestep;
    }

    if (uhd_mode)
    {
        rife_uhd_downscale_image->destroy_pipeline(flownet.opt);
        delete rife_uhd_downscale_image;

        rife_uhd_upscale_flow->destroy_pipeline(flownet.opt);
        delete rife_uhd_upscale_flow;

        rife_uhd_double_flow->destroy_pipeline(flownet.opt);
        delete rife_uhd_double_flow;
    }

    if (rife_v2)
    {
        rife_v2_slice_flow->destroy_pipeline(flownet.opt);
        delete rife_v2_slice_flow;
    }
}

#if _WIN32
static void load_param_model(ncnn::Net& net, const std::wstring& modeldir, const wchar_t* name)
{
    wchar_t parampath[256];
    wchar_t modelpath[256];
    swprintf(parampath, 256, L"%s/%s.param", modeldir.c_str(), name);
    swprintf(modelpath, 256, L"%s/%s.bin", modeldir.c_str(), name);

    {
        FILE* fp = _wfopen(parampath, L"rb");
        if (!fp)
        {
            fwprintf(stderr, L"_wfopen %ls failed\n", parampath);
        }

        net.load_param(fp);

        fclose(fp);
    }
    {
        FILE* fp = _wfopen(modelpath, L"rb");
        if (!fp)
        {
            fwprintf(stderr, L"_wfopen %ls failed\n", modelpath);
        }

        net.load_model(fp);

        fclose(fp);
    }
}
#else
static void load_param_model(ncnn::Net& net, const std::string& modeldir, const char* name)
{
    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s/%s.param", modeldir.c_str(), name);
    sprintf(modelpath, "%s/%s.bin", modeldir.c_str(), name);

    net.load_param(parampath);
    net.load_model(modelpath);
}
#endif

#if _WIN32
int RIFE::load(const std::wstring& modeldir)
#else
int RIFE::load(const std::string& modeldir)
#endif
{
    ncnn::Option opt;
    opt.num_threads = num_threads;
    opt.use_vulkan_compute = vkdev ? true : false;
    opt.use_fp16_packed = vkdev ? true : false;
    opt.use_fp16_storage = vkdev ? true : false;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = true;

    flownet.opt = opt;
    contextnet.opt = opt;
    fusionnet.opt = opt;

    flownet.set_vulkan_device(vkdev);
    contextnet.set_vulkan_device(vkdev);
    fusionnet.set_vulkan_device(vkdev);

    flownet.register_custom_layer("rife.Warp", Warp_layer_creator);
    flownet.register_custom_layer("model.gmflow.utils.split_feature", SplitFeature_layer_creator);
    flownet.register_custom_layer("model.gmflow.utils.merge_splits", MergeSplits_layer_creator);
    flownet.register_custom_layer("model.gmflow.utils.convex_upsampling", ConvexUpsampling_layer_creator);
    contextnet.register_custom_layer("rife.Warp", Warp_layer_creator);
    fusionnet.register_custom_layer("rife.Warp", Warp_layer_creator);

#if _WIN32
    load_param_model(flownet, modeldir, L"gmfss");
    if (!rife_v4)
    {
        load_param_model(contextnet, modeldir, L"contextnet");
        load_param_model(fusionnet, modeldir, L"fusionnet");
    }
#else
    load_param_model(flownet, modeldir, "flownet");
    if (!rife_v4)
    {
        load_param_model(contextnet, modeldir, "contextnet");
        load_param_model(fusionnet, modeldir, "fusionnet");
    }
#endif

    // initialize preprocess and postprocess pipeline
    if (vkdev)
    {
        std::vector<ncnn::vk_specialization_type> specializations(1);
#if _WIN32
        specializations[0].i = 1;
#else
        specializations[0].i = 0;
#endif

        {
            static std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;
            {
                ncnn::MutexLockGuard guard(lock);
                if (spirv.empty())
                {
                    if (tta_mode)
                        compile_spirv_module(rife_preproc_tta_comp_data, sizeof(rife_preproc_tta_comp_data), opt, spirv);
                    else
                        compile_spirv_module(rife_preproc_comp_data, sizeof(rife_preproc_comp_data), opt, spirv);
                }
            }

            rife_preproc = new ncnn::Pipeline(vkdev);
            rife_preproc->set_optimal_local_size_xyz(8, 8, 3);
            rife_preproc->create(spirv.data(), spirv.size() * 4, specializations);
        }

        {
            static std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;
            {
                ncnn::MutexLockGuard guard(lock);
                if (spirv.empty())
                {
                    if (tta_mode)
                        compile_spirv_module(rife_postproc_tta_comp_data, sizeof(rife_postproc_tta_comp_data), opt, spirv);
                    else
                        compile_spirv_module(rife_postproc_comp_data, sizeof(rife_postproc_comp_data), opt, spirv);
                }
            }

            rife_postproc = new ncnn::Pipeline(vkdev);
            rife_postproc->set_optimal_local_size_xyz(8, 8, 3);
            rife_postproc->create(spirv.data(), spirv.size() * 4, specializations);
        }
    }

    if (rife_v4)
    {
        if (vkdev)
        {
            static std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;
            {
                ncnn::MutexLockGuard guard(lock);
                if (spirv.empty())
                {
                    if (tta_mode)
                        compile_spirv_module(rife_v4_timestep_tta_comp_data, sizeof(rife_v4_timestep_tta_comp_data), opt, spirv);
                    else
                        compile_spirv_module(rife_v4_timestep_comp_data, sizeof(rife_v4_timestep_comp_data), opt, spirv);
                }
            }

            std::vector<ncnn::vk_specialization_type> specializations;

            rife_v4_timestep = new ncnn::Pipeline(vkdev);
            rife_v4_timestep->set_optimal_local_size_xyz(8, 8, 1);
            rife_v4_timestep->create(spirv.data(), spirv.size() * 4, specializations);
        }
    }

    return 0;
}

int RIFE::process(const ncnn::Mat& in0image, const ncnn::Mat& in1image, float timestep, ncnn::Mat& outimage) const
{
    if (!vkdev)
    {
        // cpu only
        if (rife_v4)
            return process_v4_cpu(in0image, in1image, timestep, outimage);
        /*else
            return process_cpu(in0image, in1image, timestep, outimage);*/
    }

    if (rife_v4)
        return process_v4(in0image, in1image, timestep, outimage);
    return 0;
}

int RIFE::process_v4(const ncnn::Mat& in0image, const ncnn::Mat& in1image, float timestep, ncnn::Mat& outimage) const
{
    //if (!vkdev)
    //{
    //    // cpu only
    //    return process_cpu(in0image, in1image, timestep, outimage);
    //}

    if (timestep == 0.f)
    {
        outimage = in0image;
        return 0;
    }

    if (timestep == 1.f)
    {
        outimage = in1image;
        return 0;
    }

    const unsigned char* pixel0data = (const unsigned char*)in0image.data;
    const unsigned char* pixel1data = (const unsigned char*)in1image.data;
    const int w = in0image.w;
    const int h = in0image.h;
    const int channels = 3;//in0image.elempack;

//     fprintf(stderr, "%d x %d\n", w, h);

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt = flownet.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    // pad to 32n
    int w_padded = (w + 31) / 32 * 32;
    int h_padded = (h + 31) / 32 * 32;

    const size_t in_out_tile_elemsize = opt.use_fp16_storage ? 2u : 4u;

    ncnn::Mat in0;
    ncnn::Mat in1;
    if (opt.use_fp16_storage && opt.use_int8_storage)
    {
        in0 = ncnn::Mat(w, h, (unsigned char*)pixel0data, (size_t)channels, 1);
        in1 = ncnn::Mat(w, h, (unsigned char*)pixel1data, (size_t)channels, 1);
    }
    else
    {
#if _WIN32
        in0 = ncnn::Mat::from_pixels(pixel0data, ncnn::Mat::PIXEL_BGR2RGB, w, h);
        in1 = ncnn::Mat::from_pixels(pixel1data, ncnn::Mat::PIXEL_BGR2RGB, w, h);
#else
        in0 = ncnn::Mat::from_pixels(pixel0data, ncnn::Mat::PIXEL_RGB, w, h);
        in1 = ncnn::Mat::from_pixels(pixel1data, ncnn::Mat::PIXEL_RGB, w, h);
#endif
    }

    ncnn::VkCompute cmd(vkdev);

    // upload
    ncnn::VkMat in0_gpu;
    ncnn::VkMat in1_gpu;
    {
        cmd.record_clone(in0, in0_gpu, opt);
        cmd.record_clone(in1, in1_gpu, opt);
    }

    ncnn::VkMat out_gpu;

    if (tta_mode)
    {
        // preproc
        ncnn::VkMat in0_gpu_padded[8];
        ncnn::VkMat in1_gpu_padded[8];
        ncnn::VkMat timestep_gpu_padded[2];
        {
            in0_gpu_padded[0].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in0_gpu_padded[1].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in0_gpu_padded[2].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in0_gpu_padded[3].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in0_gpu_padded[4].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in0_gpu_padded[5].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in0_gpu_padded[6].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in0_gpu_padded[7].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);

            std::vector<ncnn::VkMat> bindings(9);
            bindings[0] = in0_gpu;
            bindings[1] = in0_gpu_padded[0];
            bindings[2] = in0_gpu_padded[1];
            bindings[3] = in0_gpu_padded[2];
            bindings[4] = in0_gpu_padded[3];
            bindings[5] = in0_gpu_padded[4];
            bindings[6] = in0_gpu_padded[5];
            bindings[7] = in0_gpu_padded[6];
            bindings[8] = in0_gpu_padded[7];

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = in0_gpu.w;
            constants[1].i = in0_gpu.h;
            constants[2].i = in0_gpu.cstep;
            constants[3].i = in0_gpu_padded[0].w;
            constants[4].i = in0_gpu_padded[0].h;
            constants[5].i = in0_gpu_padded[0].cstep;

            cmd.record_pipeline(rife_preproc, bindings, constants, in0_gpu_padded[0]);
        }
        {
            in1_gpu_padded[0].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in1_gpu_padded[1].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in1_gpu_padded[2].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in1_gpu_padded[3].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in1_gpu_padded[4].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in1_gpu_padded[5].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in1_gpu_padded[6].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in1_gpu_padded[7].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);

            std::vector<ncnn::VkMat> bindings(9);
            bindings[0] = in1_gpu;
            bindings[1] = in1_gpu_padded[0];
            bindings[2] = in1_gpu_padded[1];
            bindings[3] = in1_gpu_padded[2];
            bindings[4] = in1_gpu_padded[3];
            bindings[5] = in1_gpu_padded[4];
            bindings[6] = in1_gpu_padded[5];
            bindings[7] = in1_gpu_padded[6];
            bindings[8] = in1_gpu_padded[7];

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = in1_gpu.w;
            constants[1].i = in1_gpu.h;
            constants[2].i = in1_gpu.cstep;
            constants[3].i = in1_gpu_padded[0].w;
            constants[4].i = in1_gpu_padded[0].h;
            constants[5].i = in1_gpu_padded[0].cstep;

            cmd.record_pipeline(rife_preproc, bindings, constants, in1_gpu_padded[0]);
        }
        {
            timestep_gpu_padded[0].create(w_padded, h_padded, 1, in_out_tile_elemsize, 1, blob_vkallocator);
            timestep_gpu_padded[1].create(h_padded, w_padded, 1, in_out_tile_elemsize, 1, blob_vkallocator);

            std::vector<ncnn::VkMat> bindings(2);
            bindings[0] = timestep_gpu_padded[0];
            bindings[1] = timestep_gpu_padded[1];

            std::vector<ncnn::vk_constant_type> constants(4);
            constants[0].i = timestep_gpu_padded[0].w;
            constants[1].i = timestep_gpu_padded[0].h;
            constants[2].i = timestep_gpu_padded[0].cstep;
            constants[3].f = timestep;

            cmd.record_pipeline(rife_v4_timestep, bindings, constants, timestep_gpu_padded[0]);
        }

        ncnn::VkMat out_gpu_padded[8];
        if (tta_temporal_mode)
        {
            ncnn::VkMat timestep_gpu_padded_reversed[2];
            {
                timestep_gpu_padded_reversed[0].create(w_padded, h_padded, 1, in_out_tile_elemsize, 1, blob_vkallocator);
                timestep_gpu_padded_reversed[1].create(h_padded, w_padded, 1, in_out_tile_elemsize, 1, blob_vkallocator);

                std::vector<ncnn::VkMat> bindings(2);
                bindings[0] = timestep_gpu_padded_reversed[0];
                bindings[1] = timestep_gpu_padded_reversed[1];

                std::vector<ncnn::vk_constant_type> constants(4);
                constants[0].i = timestep_gpu_padded_reversed[0].w;
                constants[1].i = timestep_gpu_padded_reversed[0].h;
                constants[2].i = timestep_gpu_padded_reversed[0].cstep;
                constants[3].f = 1.f - timestep;

                cmd.record_pipeline(rife_v4_timestep, bindings, constants, timestep_gpu_padded_reversed[0]);
            }

            ncnn::VkMat flow[4][8];
            ncnn::VkMat flow_reversed[4][8];
            for (int fi = 0; fi < 4; fi++)
            {
                for (int ti = 0; ti < 8; ti++)
                {
                    {
                        // flownet flow mask
                        ncnn::Extractor ex = flownet.create_extractor();
                        ex.set_blob_vkallocator(blob_vkallocator);
                        ex.set_workspace_vkallocator(blob_vkallocator);
                        ex.set_staging_vkallocator(staging_vkallocator);

                        ex.input("in0", in0_gpu_padded[ti]);
                        ex.input("in1", in1_gpu_padded[ti]);
                        ex.input("in2", timestep_gpu_padded[ti / 4]);

                        // intentional fall through
                        switch (fi)
                        {
                        case 3: ex.input("flow2", flow[2][ti]);
                        case 2: ex.input("flow1", flow[1][ti]);
                        case 1: ex.input("flow0", flow[0][ti]);
                        default:
                        {
                            char tmp[16];
                            sprintf(tmp, "flow%d", fi);
                            ex.extract(tmp, flow[fi][ti], cmd);
                        }
                        }
                    }

                    {
                        // flownet flow mask reversed
                        ncnn::Extractor ex = flownet.create_extractor();
                        ex.set_blob_vkallocator(blob_vkallocator);
                        ex.set_workspace_vkallocator(blob_vkallocator);
                        ex.set_staging_vkallocator(staging_vkallocator);

                        ex.input("in0", in1_gpu_padded[ti]);
                        ex.input("in1", in0_gpu_padded[ti]);
                        ex.input("in2", timestep_gpu_padded_reversed[ti / 4]);

                        // intentional fall through
                        switch (fi)
                        {
                        case 3: ex.input("flow2", flow_reversed[2][ti]);
                        case 2: ex.input("flow1", flow_reversed[1][ti]);
                        case 1: ex.input("flow0", flow_reversed[0][ti]);
                        default:
                        {
                            char tmp[16];
                            sprintf(tmp, "flow%d", fi);
                            ex.extract(tmp, flow_reversed[fi][ti], cmd);
                        }
                        }
                    }

                    // merge flow and flow_reversed
                    {
                        std::vector<ncnn::VkMat> bindings(2);
                        bindings[0] = flow[fi][ti];
                        bindings[1] = flow_reversed[fi][ti];

                        std::vector<ncnn::vk_constant_type> constants(3);
                        constants[0].i = flow[fi][ti].w;
                        constants[1].i = flow[fi][ti].h;
                        constants[2].i = flow[fi][ti].cstep;

                        ncnn::VkMat dispatcher;
                        dispatcher.w = flow[fi][ti].w;
                        dispatcher.h = flow[fi][ti].h;
                        dispatcher.c = 1;
                        cmd.record_pipeline(rife_flow_tta_temporal_avg, bindings, constants, dispatcher);
                    }
                }

                // avg flow mask
                {
                    std::vector<ncnn::VkMat> bindings(8);
                    bindings[0] = flow[fi][0];
                    bindings[1] = flow[fi][1];
                    bindings[2] = flow[fi][2];
                    bindings[3] = flow[fi][3];
                    bindings[4] = flow[fi][4];
                    bindings[5] = flow[fi][5];
                    bindings[6] = flow[fi][6];
                    bindings[7] = flow[fi][7];

                    std::vector<ncnn::vk_constant_type> constants(3);
                    constants[0].i = flow[fi][0].w;
                    constants[1].i = flow[fi][0].h;
                    constants[2].i = flow[fi][0].cstep;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = flow[fi][0].w;
                    dispatcher.h = flow[fi][0].h;
                    dispatcher.c = 1;
                    cmd.record_pipeline(rife_flow_tta_avg, bindings, constants, dispatcher);
                }
                {
                    std::vector<ncnn::VkMat> bindings(8);
                    bindings[0] = flow_reversed[fi][0];
                    bindings[1] = flow_reversed[fi][1];
                    bindings[2] = flow_reversed[fi][2];
                    bindings[3] = flow_reversed[fi][3];
                    bindings[4] = flow_reversed[fi][4];
                    bindings[5] = flow_reversed[fi][5];
                    bindings[6] = flow_reversed[fi][6];
                    bindings[7] = flow_reversed[fi][7];

                    std::vector<ncnn::vk_constant_type> constants(3);
                    constants[0].i = flow_reversed[fi][0].w;
                    constants[1].i = flow_reversed[fi][0].h;
                    constants[2].i = flow_reversed[fi][0].cstep;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = flow_reversed[fi][0].w;
                    dispatcher.h = flow_reversed[fi][0].h;
                    dispatcher.c = 1;
                    cmd.record_pipeline(rife_flow_tta_avg, bindings, constants, dispatcher);
                }
            }

            ncnn::VkMat out_gpu_padded_reversed[8];
            for (int ti = 0; ti < 8; ti++)
            {
                {
                    // flownet
                    ncnn::Extractor ex = flownet.create_extractor();
                    ex.set_blob_vkallocator(blob_vkallocator);
                    ex.set_workspace_vkallocator(blob_vkallocator);
                    ex.set_staging_vkallocator(staging_vkallocator);

                    ex.input("in0", in0_gpu_padded[ti]);
                    ex.input("in1", in1_gpu_padded[ti]);
                    ex.input("in2", timestep_gpu_padded[ti / 4]);
                    ex.input("flow0", flow[0][ti]);
                    ex.input("flow1", flow[1][ti]);
                    ex.input("flow2", flow[2][ti]);
                    ex.input("flow3", flow[3][ti]);

                    ex.extract("out0", out_gpu_padded[ti], cmd);
                }

                {
                    ncnn::Extractor ex = flownet.create_extractor();
                    ex.set_blob_vkallocator(blob_vkallocator);
                    ex.set_workspace_vkallocator(blob_vkallocator);
                    ex.set_staging_vkallocator(staging_vkallocator);

                    ex.input("in0", in1_gpu_padded[ti]);
                    ex.input("in1", in0_gpu_padded[ti]);
                    ex.input("in2", timestep_gpu_padded_reversed[ti / 4]);
                    ex.input("flow0", flow_reversed[0][ti]);
                    ex.input("flow1", flow_reversed[1][ti]);
                    ex.input("flow2", flow_reversed[2][ti]);
                    ex.input("flow3", flow_reversed[3][ti]);

                    ex.extract("out0", out_gpu_padded_reversed[ti], cmd);
                }

                // merge output
                {
                    std::vector<ncnn::VkMat> bindings(2);
                    bindings[0] = out_gpu_padded[ti];
                    bindings[1] = out_gpu_padded_reversed[ti];

                    std::vector<ncnn::vk_constant_type> constants(3);
                    constants[0].i = out_gpu_padded[ti].w;
                    constants[1].i = out_gpu_padded[ti].h;
                    constants[2].i = out_gpu_padded[ti].cstep;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = out_gpu_padded[ti].w;
                    dispatcher.h = out_gpu_padded[ti].h;
                    dispatcher.c = 3;
                    cmd.record_pipeline(rife_out_tta_temporal_avg, bindings, constants, dispatcher);
                }
            }
        }
        else
        {
            ncnn::VkMat flow[4][8];
            for (int fi = 0; fi < 4; fi++)
            {
                for (int ti = 0; ti < 8; ti++)
                {
                    // flownet flow mask
                    ncnn::Extractor ex = flownet.create_extractor();
                    ex.set_blob_vkallocator(blob_vkallocator);
                    ex.set_workspace_vkallocator(blob_vkallocator);
                    ex.set_staging_vkallocator(staging_vkallocator);

                    ex.input("in0", in0_gpu_padded[ti]);
                    ex.input("in1", in1_gpu_padded[ti]);
                    ex.input("in2", timestep_gpu_padded[ti / 4]);

                    // intentional fall through
                    switch (fi)
                    {
                    case 3: ex.input("flow2", flow[2][ti]);
                    case 2: ex.input("flow1", flow[1][ti]);
                    case 1: ex.input("flow0", flow[0][ti]);
                    default:
                    {
                        char tmp[16];
                        sprintf(tmp, "flow%d", fi);
                        ex.extract(tmp, flow[fi][ti], cmd);
                    }
                    }
                }

                // avg flow mask
                {
                    std::vector<ncnn::VkMat> bindings(8);
                    bindings[0] = flow[fi][0];
                    bindings[1] = flow[fi][1];
                    bindings[2] = flow[fi][2];
                    bindings[3] = flow[fi][3];
                    bindings[4] = flow[fi][4];
                    bindings[5] = flow[fi][5];
                    bindings[6] = flow[fi][6];
                    bindings[7] = flow[fi][7];

                    std::vector<ncnn::vk_constant_type> constants(3);
                    constants[0].i = flow[fi][0].w;
                    constants[1].i = flow[fi][0].h;
                    constants[2].i = flow[fi][0].cstep;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = flow[fi][0].w;
                    dispatcher.h = flow[fi][0].h;
                    dispatcher.c = 1;
                    cmd.record_pipeline(rife_flow_tta_avg, bindings, constants, dispatcher);
                }
            }

            for (int ti = 0; ti < 8; ti++)
            {
                // flownet
                ncnn::Extractor ex = flownet.create_extractor();
                ex.set_blob_vkallocator(blob_vkallocator);
                ex.set_workspace_vkallocator(blob_vkallocator);
                ex.set_staging_vkallocator(staging_vkallocator);

                ex.input("in0", in0_gpu_padded[ti]);
                ex.input("in1", in1_gpu_padded[ti]);
                ex.input("in2", timestep_gpu_padded[ti / 4]);
                ex.input("flow0", flow[0][ti]);
                ex.input("flow1", flow[1][ti]);
                ex.input("flow2", flow[2][ti]);
                ex.input("flow3", flow[3][ti]);

                ex.extract("out0", out_gpu_padded[ti], cmd);
            }
        }

        if (opt.use_fp16_storage && opt.use_int8_storage)
        {
            out_gpu.create(w, h, (size_t)channels, 1, blob_vkallocator);
        }
        else
        {
            out_gpu.create(w, h, channels, (size_t)4u, 1, blob_vkallocator);
        }

        // postproc
        {
            std::vector<ncnn::VkMat> bindings(9);
            bindings[0] = out_gpu_padded[0];
            bindings[1] = out_gpu_padded[1];
            bindings[2] = out_gpu_padded[2];
            bindings[3] = out_gpu_padded[3];
            bindings[4] = out_gpu_padded[4];
            bindings[5] = out_gpu_padded[5];
            bindings[6] = out_gpu_padded[6];
            bindings[7] = out_gpu_padded[7];
            bindings[8] = out_gpu;

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = out_gpu_padded[0].w;
            constants[1].i = out_gpu_padded[0].h;
            constants[2].i = out_gpu_padded[0].cstep;
            constants[3].i = out_gpu.w;
            constants[4].i = out_gpu.h;
            constants[5].i = out_gpu.cstep;

            cmd.record_pipeline(rife_postproc, bindings, constants, out_gpu);
        }
    }
    else
    {
        // preproc
        ncnn::VkMat in0_gpu_padded;
        ncnn::VkMat in1_gpu_padded;
        ncnn::VkMat timestep_gpu_padded;
        {
            in0_gpu_padded.create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);

            std::vector<ncnn::VkMat> bindings(2);
            bindings[0] = in0_gpu;
            bindings[1] = in0_gpu_padded;

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = in0_gpu.w;
            constants[1].i = in0_gpu.h;
            constants[2].i = in0_gpu.cstep;
            constants[3].i = in0_gpu_padded.w;
            constants[4].i = in0_gpu_padded.h;
            constants[5].i = in0_gpu_padded.cstep;

            cmd.record_pipeline(rife_preproc, bindings, constants, in0_gpu_padded);
        }
        {
            in1_gpu_padded.create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);

            std::vector<ncnn::VkMat> bindings(2);
            bindings[0] = in1_gpu;
            bindings[1] = in1_gpu_padded;

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = in1_gpu.w;
            constants[1].i = in1_gpu.h;
            constants[2].i = in1_gpu.cstep;
            constants[3].i = in1_gpu_padded.w;
            constants[4].i = in1_gpu_padded.h;
            constants[5].i = in1_gpu_padded.cstep;

            cmd.record_pipeline(rife_preproc, bindings, constants, in1_gpu_padded);
        }
        {
            timestep_gpu_padded.create(w_padded, h_padded, 1, in_out_tile_elemsize, 1, blob_vkallocator);

            std::vector<ncnn::VkMat> bindings(1);
            bindings[0] = timestep_gpu_padded;

            std::vector<ncnn::vk_constant_type> constants(4);
            constants[0].i = timestep_gpu_padded.w;
            constants[1].i = timestep_gpu_padded.h;
            constants[2].i = timestep_gpu_padded.cstep;
            constants[3].f = timestep;

            cmd.record_pipeline(rife_v4_timestep, bindings, constants, timestep_gpu_padded);
        }

        ncnn::VkMat out_gpu_padded;
        if (tta_temporal_mode)
        {
            ncnn::VkMat timestep_gpu_padded_reversed;
            {
                timestep_gpu_padded_reversed.create(w_padded, h_padded, 1, in_out_tile_elemsize, 1, blob_vkallocator);

                std::vector<ncnn::VkMat> bindings(1);
                bindings[0] = timestep_gpu_padded_reversed;

                std::vector<ncnn::vk_constant_type> constants(4);
                constants[0].i = timestep_gpu_padded_reversed.w;
                constants[1].i = timestep_gpu_padded_reversed.h;
                constants[2].i = timestep_gpu_padded_reversed.cstep;
                constants[3].f = 1.f - timestep;

                cmd.record_pipeline(rife_v4_timestep, bindings, constants, timestep_gpu_padded_reversed);
            }

            ncnn::VkMat flow[4];
            ncnn::VkMat flow_reversed[4];
            for (int fi = 0; fi < 4; fi++)
            {
                {
                    // flownet flow mask
                    ncnn::Extractor ex = flownet.create_extractor();
                    ex.set_blob_vkallocator(blob_vkallocator);
                    ex.set_workspace_vkallocator(blob_vkallocator);
                    ex.set_staging_vkallocator(staging_vkallocator);

                    ex.input("in0", in0_gpu_padded);
                    ex.input("in1", in1_gpu_padded);
                    ex.input("in2", timestep_gpu_padded);

                    // intentional fall through
                    switch (fi)
                    {
                    case 3: ex.input("flow2", flow[2]);
                    case 2: ex.input("flow1", flow[1]);
                    case 1: ex.input("flow0", flow[0]);
                    default:
                    {
                        char tmp[16];
                        sprintf(tmp, "flow%d", fi);
                        ex.extract(tmp, flow[fi], cmd);
                    }
                    }
                }

                {
                    // flownet flow mask reversed
                    ncnn::Extractor ex = flownet.create_extractor();
                    ex.set_blob_vkallocator(blob_vkallocator);
                    ex.set_workspace_vkallocator(blob_vkallocator);
                    ex.set_staging_vkallocator(staging_vkallocator);

                    ex.input("in0", in1_gpu_padded);
                    ex.input("in1", in0_gpu_padded);
                    ex.input("in2", timestep_gpu_padded_reversed);

                    // intentional fall through
                    switch (fi)
                    {
                    case 3: ex.input("flow2", flow_reversed[2]);
                    case 2: ex.input("flow1", flow_reversed[1]);
                    case 1: ex.input("flow0", flow_reversed[0]);
                    default:
                    {
                        char tmp[16];
                        sprintf(tmp, "flow%d", fi);
                        ex.extract(tmp, flow_reversed[fi], cmd);
                    }
                    }
                }

                // merge flow and flow_reversed
                {
                    std::vector<ncnn::VkMat> bindings(2);
                    bindings[0] = flow[fi];
                    bindings[1] = flow_reversed[fi];

                    std::vector<ncnn::vk_constant_type> constants(3);
                    constants[0].i = flow[fi].w;
                    constants[1].i = flow[fi].h;
                    constants[2].i = flow[fi].cstep;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = flow[fi].w;
                    dispatcher.h = flow[fi].h;
                    dispatcher.c = 1;
                    cmd.record_pipeline(rife_flow_tta_temporal_avg, bindings, constants, dispatcher);
                }
            }

            {
                // flownet
                ncnn::Extractor ex = flownet.create_extractor();
                ex.set_blob_vkallocator(blob_vkallocator);
                ex.set_workspace_vkallocator(blob_vkallocator);
                ex.set_staging_vkallocator(staging_vkallocator);

                ex.input("in0", in0_gpu_padded);
                ex.input("in1", in1_gpu_padded);
                ex.input("in2", timestep_gpu_padded);
                ex.input("flow0", flow[0]);
                ex.input("flow1", flow[1]);
                ex.input("flow2", flow[2]);
                ex.input("flow3", flow[3]);

                ex.extract("out0", out_gpu_padded, cmd);
            }

            ncnn::VkMat out_gpu_padded_reversed;
            {
                ncnn::Extractor ex = flownet.create_extractor();
                ex.set_blob_vkallocator(blob_vkallocator);
                ex.set_workspace_vkallocator(blob_vkallocator);
                ex.set_staging_vkallocator(staging_vkallocator);

                ex.input("in0", in1_gpu_padded);
                ex.input("in1", in0_gpu_padded);
                ex.input("in2", timestep_gpu_padded_reversed);
                ex.input("flow0", flow_reversed[0]);
                ex.input("flow1", flow_reversed[1]);
                ex.input("flow2", flow_reversed[2]);
                ex.input("flow3", flow_reversed[3]);

                ex.extract("out0", out_gpu_padded_reversed, cmd);
            }

            // merge output
            {
                std::vector<ncnn::VkMat> bindings(2);
                bindings[0] = out_gpu_padded;
                bindings[1] = out_gpu_padded_reversed;

                std::vector<ncnn::vk_constant_type> constants(3);
                constants[0].i = out_gpu_padded.w;
                constants[1].i = out_gpu_padded.h;
                constants[2].i = out_gpu_padded.cstep;

                ncnn::VkMat dispatcher;
                dispatcher.w = out_gpu_padded.w;
                dispatcher.h = out_gpu_padded.h;
                dispatcher.c = 3;
                cmd.record_pipeline(rife_out_tta_temporal_avg, bindings, constants, dispatcher);
            }
        }
        else
        {
            // flownet
            ncnn::Extractor ex = flownet.create_extractor();
            ex.set_blob_vkallocator(blob_vkallocator);
            ex.set_workspace_vkallocator(blob_vkallocator);
            ex.set_staging_vkallocator(staging_vkallocator);

            ex.input("in0", in0_gpu_padded);
            ex.input("in1", in1_gpu_padded);
            ex.input("in2", timestep_gpu_padded);
            ex.extract("out0", out_gpu_padded, cmd);
        }

        if (opt.use_fp16_storage && opt.use_int8_storage)
        {
            out_gpu.create(w, h, (size_t)channels, 1, blob_vkallocator);
        }
        else
        {
            out_gpu.create(w, h, channels, (size_t)4u, 1, blob_vkallocator);
        }

        // postproc
        {
            std::vector<ncnn::VkMat> bindings(2);
            bindings[0] = out_gpu_padded;
            bindings[1] = out_gpu;

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = out_gpu_padded.w;
            constants[1].i = out_gpu_padded.h;
            constants[2].i = out_gpu_padded.cstep;
            constants[3].i = out_gpu.w;
            constants[4].i = out_gpu.h;
            constants[5].i = out_gpu.cstep;

            cmd.record_pipeline(rife_postproc, bindings, constants, out_gpu);
        }
    }

    // download
    {
        ncnn::Mat out;

        if (opt.use_fp16_storage && opt.use_int8_storage)
        {
            out = ncnn::Mat(out_gpu.w, out_gpu.h, (unsigned char*)outimage.data, (size_t)channels, 1);
        }

        cmd.record_clone(out_gpu, out, opt);

        cmd.submit_and_wait();

        if (!(opt.use_fp16_storage && opt.use_int8_storage))
        {
#if _WIN32
            out.to_pixels((unsigned char*)outimage.data, ncnn::Mat::PIXEL_RGB2BGR);
#else
            out.to_pixels((unsigned char*)outimage.data, ncnn::Mat::PIXEL_RGB);
#endif
        }
    }

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    return 0;
}

int RIFE::process_v4_cpu(const ncnn::Mat& in0image, const ncnn::Mat& in1image, float timestep, ncnn::Mat& outimage) const
{
    if (timestep == 0.f)
    {
        outimage = in0image;
        return 0;
    }

    if (timestep == 1.f)
    {
        outimage = in1image;
        return 0;
    }

    const unsigned char* pixel0data = (const unsigned char*)in0image.data;
    const unsigned char* pixel1data = (const unsigned char*)in1image.data;
    const int w = in0image.w;
    const int h = in0image.h;
    const int channels = 3;//in0image.elempack;

//     fprintf(stderr, "%d x %d\n", w, h);

    ncnn::Option opt = flownet.opt;

    // pad to 32n
    int w_padded = (w + 31) / 32 * 32;
    int h_padded = (h + 31) / 32 * 32;

    ncnn::Mat in0;
    ncnn::Mat in1;
    {
#if _WIN32
        in0 = ncnn::Mat::from_pixels(pixel0data, ncnn::Mat::PIXEL_BGR2RGB, w, h);
        in1 = ncnn::Mat::from_pixels(pixel1data, ncnn::Mat::PIXEL_BGR2RGB, w, h);
#else
        in0 = ncnn::Mat::from_pixels(pixel0data, ncnn::Mat::PIXEL_RGB, w, h);
        in1 = ncnn::Mat::from_pixels(pixel1data, ncnn::Mat::PIXEL_RGB, w, h);
#endif
    }

    ncnn::Mat flow;

    {
        // preproc and border padding
        ncnn::Mat in0_padded;
        ncnn::Mat in1_padded;
        ncnn::Mat timestep_padded;

        
        float mean_vals[3] = { 0.485, 0.456, 0.406};
        float norm_vals[3] = { 0.229, 0.224, 0.225};
        {
            in0_padded.create(w_padded, h_padded, 3);
            for (int q = 0; q < 3; q++)
            {
                float* outptr = in0_padded.channel(q);

                int i = 0;
                for (; i < h; i++)
                {
                    const float* ptr = in0.channel(q).row(i);

                    int j = 0;
                    for (; j < w; j++)
                    {
                        *outptr++ = (*ptr++ * (1 / 255.f) - mean_vals[q]) / norm_vals[q];
                    }
                    for (; j < w_padded; j++)
                    {
                        *outptr++ = 0.f;
                    }
                }
                for (; i < h_padded; i++)
                {
                    for (int j = 0; j < w_padded; j++)
                    {
                        *outptr++ = 0.f;
                    }
                }
            }
        }
        {
            in1_padded.create(w_padded, h_padded, 3);
            for (int q = 0; q < 3; q++)
            {
                float* outptr = in1_padded.channel(q);

                int i = 0;
                for (; i < h; i++)
                {
                    const float* ptr = in1.channel(q).row(i);

                    int j = 0;
                    for (; j < w; j++)
                    {
                        *outptr++ = (*ptr++ * (1 / 255.f) - mean_vals[q]) / norm_vals[q];
                    }
                    for (; j < w_padded; j++)
                    {
                        *outptr++ = 0.f;
                    }
                }
                for (; i < h_padded; i++)
                {
                    for (int j = 0; j < w_padded; j++)
                    {
                        *outptr++ = 0.f;
                    }
                }
            }
        }
        {
            timestep_padded.create(w_padded, h_padded, 1);
            timestep_padded.fill(timestep);
        }

        ncnn::Mat flow_padded;
        ncnn::Mat out_padded_reversed;

        print_mat(in0_padded, "in0_padded");

        // Some Test
        // ncnn::Mat testMat, testOutMat;
        // testMat.create(6, 4, 2);
        // //testOutMat.create(3, 2, 2, 4);
        // int test_cnt = 0;
        // float* testMat_ptr = (float *)testMat.data;
        // for(int c=0;c<2;c++)
        // for(int i = 0; i < 6; i++)
        // {
        //     for(int j = 0; j < 4; j++)
        //     {
        //         *(testMat_ptr++) = test_cnt * 1.f;
        //         test_cnt++;
        //     }
        // }
        
        // ncnn::Layer* op = SplitFeature_layer_creator(nullptr);

        // // set param
        // ncnn::ParamDict pd;
        // pd.set(0, 2);// op_type

        // op->load_param(pd);

        // op->create_pipeline(opt);
        // op->forward(testMat, testOutMat, opt);
        // op->destroy_pipeline(opt);
        // delete op;
        // print_mat(testOutMat, "test_out");

        {
            // flownet
            ncnn::Extractor ex = flownet.create_extractor();

            ex.input("in0", in0_padded);
            ex.input("in1", in1_padded);
            // ex.input("in2", timestep_padded);
            ex.extract("out0", flow_padded);
        }

        // cut padding and postproc
        flow.create(w, h, 2);
        print_mat(flow_padded, "flow_padded");
    }

    // download
    {
#if _WIN32
        flow.to_pixels((unsigned char*)outimage.data, ncnn::Mat::PIXEL_GRAY2RGB);
#else
        out.to_pixels((unsigned char*)outimage.data, ncnn::Mat::PIXEL_RGB);
#endif
    }

    // save to file

    return 0;
}
