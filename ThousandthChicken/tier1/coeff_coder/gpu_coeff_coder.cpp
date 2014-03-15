
#include "gpu_coder.h"

#include "../../print_info/print_info.h"

#include <iostream>
#include <string>
#include <fstream>

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include <list>


#include "gpu_coeff_coder2.h"

float gpuDecode(EntropyCodingTaskInfo *infos, int count)
{

	int codeBlocks = count;
	int maxOutLength = MAX_CODESTREAM_SIZE;

	int n = 0;
	for(int i = 0; i < codeBlocks; i++)
		n += infos[i].width * infos[i].height;

	byte *d_inbuf;
	GPU_JPEG2K::CoefficientState *d_stBuffors;


	CodeBlockAdditionalInfo *h_infos = (CodeBlockAdditionalInfo *) malloc(sizeof(CodeBlockAdditionalInfo) * codeBlocks);
	CodeBlockAdditionalInfo *d_infos;



#ifdef CUDA



	cuda_d_allocate_mem((void **) &d_inbuf, sizeof(byte) * codeBlocks * maxOutLength);
	cuda_d_allocate_mem((void **) &d_infos, sizeof(CodeBlockAdditionalInfo) * codeBlocks);

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

		cuda_d_allocate_mem((void **) &(h_infos[i].coefficients), sizeof(int) * infos[i].nominalWidth * infos[i].nominalHeight);
		infos[i].coefficients = h_infos[i].coefficients;

		cuda_memcpy_htd(infos[i].codeStream, (void *) (d_inbuf + i * maxOutLength), sizeof(byte) * infos[i].length);

		magconOffset += h_infos[i].width * (h_infos[i].stripeNo + 2);
	}

	cuda_d_allocate_mem((void **) &d_stBuffors, sizeof(GPU_JPEG2K::CoefficientState) * magconOffset);
	CHECK_ERRORS(cudaMemset((void *) d_stBuffors, 0, sizeof(GPU_JPEG2K::CoefficientState) * magconOffset));

	cuda_memcpy_htd(h_infos, d_infos, sizeof(CodeBlockAdditionalInfo) * codeBlocks);

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start, 0);

	CHECK_ERRORS(GPU_JPEG2K::launch_decode((int) ceil((float) codeBlocks / THREADS), THREADS, d_stBuffors, d_inbuf, maxOutLength, d_infos, codeBlocks));

	cudaEventRecord(end, 0);

	cuda_d_free(d_inbuf);
	cuda_d_free(d_stBuffors);
	cuda_d_free(d_infos);

	free(h_infos);

	float elapsed = 0.0f;
	cudaEventElapsedTime(&elapsed, start, end);
	
	return elapsed;
#endif
	return 0;
}

void extract_cblks(type_tile *tile, std::list<type_codeblock *> &out_cblks)
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
				for(uint32_t l = 0; l < sb->num_cblks; l++)
					out_cblks.push_back(&(sb->cblks[l]));
			}
		}
	}
}
void convert_to_decoding_task(EntropyCodingTaskInfo &task, const type_codeblock &cblk)
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

	//task.coefficients = cblk.data_d;
}

void decode_tile(type_tile *tile)
{
//	println_start(INFO);

//	start_measure();

	std::list<type_codeblock *> cblks;
	extract_cblks(tile, cblks);

	EntropyCodingTaskInfo *tasks = (EntropyCodingTaskInfo *) malloc(sizeof(EntropyCodingTaskInfo) * cblks.size());

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
