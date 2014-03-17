#include "CoefficientCoderGPU.h"

#include "stdafx.h"

#include <iostream>
#include "basic.hpp"
#include "cmdparser.hpp"
#include "oclobject.hpp"
#include "utils.h"

// for perf. counters
#include <Windows.h>

#include <iostream>
#include <string>


using namespace std;
#include "tier1/coeff_coder/gpu_coeff_coder2.h"


void CoefficientCoderGPU::decode_tile(type_tile *tile)
{
//	println_start(INFO);

//	start_measure();

	std::list<type_codeblock *> cblks;
	extract_cblks(tile, cblks);

	EntropyCodingTaskInfo *tasks = (EntropyCodingTaskInfo *) calloc( cblks.size(),sizeof(EntropyCodingTaskInfo));

	std::list<type_codeblock *>::iterator ii = cblks.begin();

	int num_tasks = 0;
	for(; ii != cblks.end(); ++ii)
	{
		convert_to_decoding_task(tasks[num_tasks++], *(*ii));
	}

//	printf("%d\n", num_tasks);

	float t = gpuDecode(tasks, num_tasks);

	printf("kernel consumption: %f\n", t);

	ii = cblks.begin();

	for(int i = 0; i < num_tasks; i++, ++ii)
	{
		(*ii)->data_d = tasks[i].coefficients;
	}

	free(tasks);

//	stop_measure(INFO);

//	println_end(INFO);
}

void CoefficientCoderGPU::extract_cblks(type_tile *tile, std::list<type_codeblock *> &out_cblks)
{
	for(int i = 0; i < tile->parent_img->num_components; i++)
	{
		type_tile_comp *tile_comp = &(tile->tile_comp[i]);
		for(int j = 0; j < tile_comp->num_rlvls; j++)
		{
			type_res_lvl *res_lvl = &(tile_comp->res_lvls[j]);
			for(int k = 0; k < res_lvl->num_subbands; k++)
			{
				type_subband *sb = &(res_lvl->subbands[k]);
				for(unsigned int l = 0; l < sb->num_cblks; l++)
					out_cblks.push_back(&(sb->cblks[l]));
			}
		}
	}
}
void CoefficientCoderGPU::convert_to_decoding_task(EntropyCodingTaskInfo &task, const type_codeblock &cblk)
{
	switch(cblk.parent_sb->orient)
	{
	case LL:
	case LH:
		task.subband = 0;
		break;
	case HL:
		task.subband = 1;
		break;
	case HH:
		task.subband = 2;
		break;
	}

	task.width = cblk.width;
	task.height = cblk.height;

	task.nominalWidth = cblk.parent_sb->parent_res_lvl->parent_tile_comp->cblk_w;
	task.nominalHeight = cblk.parent_sb->parent_res_lvl->parent_tile_comp->cblk_h;

	task.magbits = cblk.parent_sb->mag_bits;

	task.codeStream = cblk.codestream;
	task.length = cblk.length;
	task.significantBits = cblk.significant_bits;
	task.coefficients = NULL;
}

float CoefficientCoderGPU::gpuDecode(EntropyCodingTaskInfo *infos, int count)
{

    LARGE_INTEGER perf_freq;
    LARGE_INTEGER perf_start;
    LARGE_INTEGER perf_stop;

    cl_int err = CL_SUCCESS;

	int codeBlocks = count;
	int maxOutLength = MAX_CODESTREAM_SIZE;


    // execute kernel
    QueryPerformanceCounter(&perf_start);

    //allocate d_inbuf on device
	cl_mem d_inbuf = clCreateBuffer(oclObjects.context, CL_MEM_READ_WRITE, sizeof(unsigned char) * codeBlocks * maxOutLength, NULL, &err);
    SAMPLE_CHECK_ERRORS(err);
    if (d_inbuf == (cl_mem)0)
        throw Error("Failed to create d_inbuf Buffer!");

	// allocate and initialize h_infos on host
	cl_uint dev_alignment = requiredOpenCLAlignment(oclObjects.device);
	CodeBlockAdditionalInfo *h_infos = (CodeBlockAdditionalInfo *) aligned_malloc(sizeof(CodeBlockAdditionalInfo) * codeBlocks,dev_alignment);
	int magconOffset = 0;
	for(int i = 0; i < codeBlocks; i++)
	{
		h_infos[i].width = infos[i].width;
		h_infos[i].height = infos[i].height;
		h_infos[i].nominalWidth = infos[i].nominalWidth;
		h_infos[i].stripeNo = (int) ceil(infos[i].height / 4.0f);
		h_infos[i].subband = infos[i].subband;
		h_infos[i].magconOffset = magconOffset + infos[i].width;
		h_infos[i].magbits = infos[i].magbits;
		h_infos[i].length = infos[i].length;
		h_infos[i].significantBits = infos[i].significantBits;
		h_infos[i].coefficients = NULL;

		//allocate coefficients buffer on device, and copy to h_infos[i].coefficients and infos[i].coefficients
#ifdef CUDA
		cuda_d_allocate_mem((void **) &(h_infos[i].coefficients), sizeof(int) * infos[i].nominalWidth * infos[i].nominalHeight);
		infos[i].coefficients = h_infos[i].coefficients;
#endif

	    //copy codestream buffer to device
#ifdef CUDA
		cuda_memcpy_htd(infos[i].codeStream, (void *) (d_inbuf + i * maxOutLength), sizeof(unsigned char) * infos[i].length);
#endif

		magconOffset += h_infos[i].width * (h_infos[i].stripeNo + 2);
	}

	//allocate d_stBuffers on device and initialize it to zero
	cl_mem d_stBuffers = clCreateBuffer(oclObjects.context, CL_MEM_READ_WRITE ,  sizeof(unsigned int) * magconOffset, NULL, &err);
    SAMPLE_CHECK_ERRORS(err);
    if (d_stBuffers == (cl_mem)0)
        throw Error("Failed to create d_infos Buffer!");
	cl_int pattern = 0;
	clEnqueueFillBuffer(oclObjects.queue, d_stBuffers, &pattern, sizeof(cl_int), 0, sizeof(unsigned int) * magconOffset, 0, NULL, NULL);


    //allocate and initialize d_infos on device
	cl_mem d_infos = clCreateBuffer(oclObjects.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,  sizeof(CodeBlockAdditionalInfo) * codeBlocks, h_infos, &err);
    SAMPLE_CHECK_ERRORS(err);
    if (d_infos == (cl_mem)0)
        throw Error("Failed to create d_infos Buffer!");

	/////////////////////////////////////////////////////////////////////////////
	// set kernel arguments ///////////////////////////////
	
	//set kernel argument 0 to d_stBuffers
	err = clSetKernelArg(executable.kernel, 0, sizeof(cl_mem), (void *) &d_stBuffers);
    SAMPLE_CHECK_ERRORS(err);
	
	//set kernel argument 1 to d_inbuf
	err = clSetKernelArg(executable.kernel, 1, sizeof(cl_mem), (void *) &d_inbuf);
    SAMPLE_CHECK_ERRORS(err);
	
	//set kernel argument 2 to maxOutLength
	err = clSetKernelArg(executable.kernel, 2, sizeof(int), (void *) &maxOutLength);
    SAMPLE_CHECK_ERRORS(err);

	//set kernel argument 3 to d_infos
	err = clSetKernelArg(executable.kernel, 3, sizeof(cl_mem), (void *) &d_infos);
    SAMPLE_CHECK_ERRORS(err);

	//set kernel argument 4 to codeBlocks
	err = clSetKernelArg(executable.kernel, 4, sizeof(int), (void *) &codeBlocks);
    SAMPLE_CHECK_ERRORS(err);
	//////////////////////////////////////////////////////////////////////////////////

	
	//decode
	//CHECK_ERRORS(GPU_JPEG2K::launch_decode((int) ceil((float) codeBlocks / THREADS), THREADS, d_stBuffors, d_inbuf, maxOutLength, d_infos, codeBlocks));
	//__kernel void g_decode(__global unsigned int *coeffBuffers, __global unsigned char *inbuf, int maxThreadBufferLength, __global CodeBlockAdditionalInfo *infos, int codeBlocks)

	size_t global_work_size[1] = { codeBlocks};
    // execute kernel
    err = clEnqueueNDRangeKernel(oclObjects.queue, executable.kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);

	err = clFinish(oclObjects.queue);
    SAMPLE_CHECK_ERRORS(err);
    QueryPerformanceCounter(&perf_stop);

	 err = clReleaseMemObject(d_infos);
    SAMPLE_CHECK_ERRORS(err);

	err = clReleaseMemObject(d_stBuffers);
    SAMPLE_CHECK_ERRORS(err);
		
	err = clReleaseMemObject(d_inbuf);
    SAMPLE_CHECK_ERRORS(err);

	aligned_free(h_infos);

	// retrieve perf. counter frequency
	QueryPerformanceCounter(&perf_stop);
    QueryPerformanceFrequency(&perf_freq);
    return (float)(perf_stop.QuadPart - perf_start.QuadPart)/(float)perf_freq.QuadPart;

}

#ifdef CUDA
void launch_decode(dim3 gridDim, dim3 blockDim, unsigned int *coeffBuffers, unsigned char *inbuf, int maxThreadBufforLength, CodeBlockAdditionalInfo *infos, int codeBlocks)
{
	g_decode<<<gridDim, blockDim>>>(coeffBuffers, inbuf, maxThreadBufforLength, infos, codeBlocks);
}
#endif
