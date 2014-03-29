#pragma once


#include "tier1/coeff_coder/gpu_coder_basic.h"
#include <list>
#include "stdafx.h"

#include <iostream>
#include "basic.hpp"
#include "cmdparser.hpp"
#include "oclobject.hpp"

class CoefficientCoderGPU
{
public:
	CoefficientCoderGPU(OpenCLBasic& oclObjects) : oclObjects(oclObjects),
		                                           executable(oclObjects,"CoefficientCoder.cl","","g_decode",
												   "-I C:\\src\\ThousandthChicken\\ThousandthChicken\\tier1\\coeff_coder")
												   // -g -s \"c:\\src\\ThousandthChicken\\ThousandthChicken\\CoefficientCoder.cl\"
	{}
	void decode_tile(type_tile *tile);
private:
	float gpuDecode(EntropyCodingTaskInfo *infos, int count);
	void convert_to_decoding_task(EntropyCodingTaskInfo &task, const type_codeblock &cblk);
	void extract_cblks(type_tile *tile, std::list<type_codeblock *> &out_cblks);
	OpenCLBasic& oclObjects;
	OpenCLProgramOneKernel executable;
};

