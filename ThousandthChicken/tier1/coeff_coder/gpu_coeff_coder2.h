#pragma once

#define THREADS 32

#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __constant__
#define __shared__
#endif


typedef struct
{
	int length;
	unsigned char significantBits;
	unsigned char codingPasses;
	unsigned char width;
	unsigned char nominalWidth;
	unsigned char height;
	unsigned char stripeNo;
	unsigned char magbits;
	unsigned char subband;
	unsigned char compType;
	unsigned char dwtLevel;
	float stepSize;

	int magconOffset;

	int* coefficients;
} CodeBlockAdditionalInfo;

namespace GPU_JPEG2K
{
	typedef unsigned int CoefficientState;
	typedef unsigned char byte;
#ifdef CUDA
	void launch_decode(dim3 gridDim, dim3 blockDim, CoefficientState *coeffBuffors, byte *inbuf, int maxThreadBufforLength, CodeBlockAdditionalInfo *infos, int codeBlocks);
#endif	
}

