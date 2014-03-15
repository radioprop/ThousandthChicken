#ifdef DEBUG_MQ
__device__ int l = 0;
__device__ int Cstates[1200000];
#endif

#include "gpu_coeff_coder2.h"

#define __global__
#define __device__
#define __shared__


namespace GPU_JPEG2K
{
	#include "gpu_mq-coder.h"

__device__ void SetMaskedBits(unsigned int &reg, unsigned int mask, unsigned int bits)
{
	reg = (reg & ~mask) | (bits & mask);
}

__device__ void SetNthBit(unsigned int &reg, unsigned int n)
{
	SetMaskedBits(reg, 1 << n, 1 << n);
}


typedef struct
{
	CoefficientState tl;
	CoefficientState t;
	CoefficientState tr;
	
	CoefficientState l;
	CoefficientState c;
	CoefficientState r;
	
	CoefficientState bl;
	CoefficientState b;
	CoefficientState br;

	short pos;
} CtxWindow;

__device__ void down(CodeBlockAdditionalInfo &info, CtxWindow &window, CoefficientState *coeffs)
{
	window.tr = coeffs[window.pos + 1 - info.width];
	window.r = coeffs[window.pos + 1];
	window.br = coeffs[window.pos + 1 + info.width];
}

__device__ void up(CtxWindow &window, CoefficientState *coeffs)
{
	coeffs[window.pos - 1] = window.l;
}

__device__ void shift(CtxWindow &window)
{
	window.tl = window.t; window.t = window.tr; window.tr = 0; // top layer
	window.l = window.c; window.c = window.r; window.r = 0; // middle layer
	window.bl = window.b; window.b = window.br; window.br = 0; // bottom layer
	window.pos += 1;
}

typedef int CtxReg;

#define TRIMASK 0x249 //((1 << 0) | (1 << 3) | (1 << 6) | (1 << 9))

__device__ CtxReg buildCtxReg(CtxWindow &window, unsigned char bitoffset)
{
	CtxReg reg = 0;

	reg |= ((window.tl >> (bitoffset + 9)) & 1) << 0;
	reg |= ((window.t >> (bitoffset + 9)) & 1) << 1;
	reg |= ((window.tr >> (bitoffset + 9)) & 1) << 2;
	reg |= ((window.l >> (bitoffset)) & TRIMASK) << 3;
	reg |= ((window.c >> (bitoffset)) & TRIMASK) << 4;
	reg |= ((window.r >> (bitoffset)) & TRIMASK) << 5;
	reg |= ((window.bl >> (bitoffset)) & 1) << 15;
	reg |= ((window.b >> (bitoffset)) & 1) << 16;
	reg |= ((window.br >> (bitoffset)) & 1) << 17;
	
	return reg;
}

	__constant__ unsigned char SPCXLUT[3][512] = {
		{
			0, 1, 3, 3, 1, 2, 3, 3, 5, 6, 7, 7, 6, 6, 7, 7, 0, 1, 3, 3,
			1, 2, 3, 3, 5, 6, 7, 7, 6, 6, 7, 7, 5, 6, 7, 7, 6, 6, 7, 7,
			8, 8, 8, 8, 8, 8, 8, 8, 5, 6, 7, 7, 6, 6, 7, 7, 8, 8, 8, 8,
			8, 8, 8, 8, 1, 2, 3, 3, 2, 2, 3, 3, 6, 6, 7, 7, 6, 6, 7, 7,
			1, 2, 3, 3, 2, 2, 3, 3, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7,
			6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 7, 7, 6, 6, 7, 7,
			8, 8, 8, 8, 8, 8, 8, 8, 3, 3, 4, 4, 3, 3, 4, 4, 7, 7, 7, 7,
			7, 7, 7, 7, 3, 3, 4, 4, 3, 3, 4, 4, 7, 7, 7, 7, 7, 7, 7, 7,
			7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7,
			7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 3, 3, 4, 4, 3, 3, 4, 4,
			7, 7, 7, 7, 7, 7, 7, 7, 3, 3, 4, 4, 3, 3, 4, 4, 7, 7, 7, 7,
			7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8,
			7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 1, 2, 3, 3,
			2, 2, 3, 3, 6, 6, 7, 7, 6, 6, 7, 7, 1, 2, 3, 3, 2, 2, 3, 3,
			6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7, 8, 8, 8, 8,
			8, 8, 8, 8, 6, 6, 7, 7, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8,
			2, 2, 3, 3, 2, 2, 3, 3, 6, 6, 7, 7, 6, 6, 7, 7, 2, 2, 3, 3,
			2, 2, 3, 3, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7,
			8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 7, 7, 6, 6, 7, 7, 8, 8, 8, 8,
			8, 8, 8, 8, 3, 3, 4, 4, 3, 3, 4, 4, 7, 7, 7, 7, 7, 7, 7, 7,
			3, 3, 4, 4, 3, 3, 4, 4, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
			7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7,
			8, 8, 8, 8, 8, 8, 8, 8, 3, 3, 4, 4, 3, 3, 4, 4, 7, 7, 7, 7,
			7, 7, 7, 7, 3, 3, 4, 4, 3, 3, 4, 4, 7, 7, 7, 7, 7, 7, 7, 7,
			7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7,
			7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8
		},
		{
			0, 1, 5, 6, 1, 2, 6, 6, 3, 3, 7, 7, 3, 3, 7, 7, 0, 1, 5, 6,
			1, 2, 6, 6, 3, 3, 7, 7, 3, 3, 7, 7, 3, 3, 7, 7, 3, 3, 7, 7,
			4, 4, 7, 7, 4, 4, 7, 7, 3, 3, 7, 7, 3, 3, 7, 7, 4, 4, 7, 7,
			4, 4, 7, 7, 1, 2, 6, 6, 2, 2, 6, 6, 3, 3, 7, 7, 3, 3, 7, 7,
			1, 2, 6, 6, 2, 2, 6, 6, 3, 3, 7, 7, 3, 3, 7, 7, 3, 3, 7, 7,
			3, 3, 7, 7, 4, 4, 7, 7, 4, 4, 7, 7, 3, 3, 7, 7, 3, 3, 7, 7,
			4, 4, 7, 7, 4, 4, 7, 7, 5, 6, 8, 8, 6, 6, 8, 8, 7, 7, 8, 8,
			7, 7, 8, 8, 5, 6, 8, 8, 6, 6, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8,
			7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8,
			7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 6, 6, 8, 8, 6, 6, 8, 8,
			7, 7, 8, 8, 7, 7, 8, 8, 6, 6, 8, 8, 6, 6, 8, 8, 7, 7, 8, 8,
			7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8,
			7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 1, 2, 6, 6,
			2, 2, 6, 6, 3, 3, 7, 7, 3, 3, 7, 7, 1, 2, 6, 6, 2, 2, 6, 6,
			3, 3, 7, 7, 3, 3, 7, 7, 3, 3, 7, 7, 3, 3, 7, 7, 4, 4, 7, 7,
			4, 4, 7, 7, 3, 3, 7, 7, 3, 3, 7, 7, 4, 4, 7, 7, 4, 4, 7, 7,
			2, 2, 6, 6, 2, 2, 6, 6, 3, 3, 7, 7, 3, 3, 7, 7, 2, 2, 6, 6,
			2, 2, 6, 6, 3, 3, 7, 7, 3, 3, 7, 7, 3, 3, 7, 7, 3, 3, 7, 7,
			4, 4, 7, 7, 4, 4, 7, 7, 3, 3, 7, 7, 3, 3, 7, 7, 4, 4, 7, 7,
			4, 4, 7, 7, 6, 6, 8, 8, 6, 6, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8,
			6, 6, 8, 8, 6, 6, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8,
			7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8,
			7, 7, 8, 8, 7, 7, 8, 8, 6, 6, 8, 8, 6, 6, 8, 8, 7, 7, 8, 8,
			7, 7, 8, 8, 6, 6, 8, 8, 6, 6, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8,
			7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8,
			7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8
		},
		{
			0, 3, 1, 4, 3, 6, 4, 7, 1, 4, 2, 5, 4, 7, 5, 7, 0, 3, 1, 4,
			3, 6, 4, 7, 1, 4, 2, 5, 4, 7, 5, 7, 1, 4, 2, 5, 4, 7, 5, 7,
			2, 5, 2, 5, 5, 7, 5, 7, 1, 4, 2, 5, 4, 7, 5, 7, 2, 5, 2, 5,
			5, 7, 5, 7, 3, 6, 4, 7, 6, 8, 7, 8, 4, 7, 5, 7, 7, 8, 7, 8,
			3, 6, 4, 7, 6, 8, 7, 8, 4, 7, 5, 7, 7, 8, 7, 8, 4, 7, 5, 7,
			7, 8, 7, 8, 5, 7, 5, 7, 7, 8, 7, 8, 4, 7, 5, 7, 7, 8, 7, 8,
			5, 7, 5, 7, 7, 8, 7, 8, 1, 4, 2, 5, 4, 7, 5, 7, 2, 5, 2, 5,
			5, 7, 5, 7, 1, 4, 2, 5, 4, 7, 5, 7, 2, 5, 2, 5, 5, 7, 5, 7,
			2, 5, 2, 5, 5, 7, 5, 7, 2, 5, 2, 5, 5, 7, 5, 7, 2, 5, 2, 5,
			5, 7, 5, 7, 2, 5, 2, 5, 5, 7, 5, 7, 4, 7, 5, 7, 7, 8, 7, 8,
			5, 7, 5, 7, 7, 8, 7, 8, 4, 7, 5, 7, 7, 8, 7, 8, 5, 7, 5, 7,
			7, 8, 7, 8, 5, 7, 5, 7, 7, 8, 7, 8, 5, 7, 5, 7, 7, 8, 7, 8,
			5, 7, 5, 7, 7, 8, 7, 8, 5, 7, 5, 7, 7, 8, 7, 8, 3, 6, 4, 7,
			6, 8, 7, 8, 4, 7, 5, 7, 7, 8, 7, 8, 3, 6, 4, 7, 6, 8, 7, 8,
			4, 7, 5, 7, 7, 8, 7, 8, 4, 7, 5, 7, 7, 8, 7, 8, 5, 7, 5, 7,
			7, 8, 7, 8, 4, 7, 5, 7, 7, 8, 7, 8, 5, 7, 5, 7, 7, 8, 7, 8,
			6, 8, 7, 8, 8, 8, 8, 8, 7, 8, 7, 8, 8, 8, 8, 8, 6, 8, 7, 8,
			8, 8, 8, 8, 7, 8, 7, 8, 8, 8, 8, 8, 7, 8, 7, 8, 8, 8, 8, 8,
			7, 8, 7, 8, 8, 8, 8, 8, 7, 8, 7, 8, 8, 8, 8, 8, 7, 8, 7, 8,
			8, 8, 8, 8, 4, 7, 5, 7, 7, 8, 7, 8, 5, 7, 5, 7, 7, 8, 7, 8,
			4, 7, 5, 7, 7, 8, 7, 8, 5, 7, 5, 7, 7, 8, 7, 8, 5, 7, 5, 7,
			7, 8, 7, 8, 5, 7, 5, 7, 7, 8, 7, 8, 5, 7, 5, 7, 7, 8, 7, 8,
			5, 7, 5, 7, 7, 8, 7, 8, 7, 8, 7, 8, 8, 8, 8, 8, 7, 8, 7, 8,
			8, 8, 8, 8, 7, 8, 7, 8, 8, 8, 8, 8, 7, 8, 7, 8, 8, 8, 8, 8,
			7, 8, 7, 8, 8, 8, 8, 8, 7, 8, 7, 8, 8, 8, 8, 8, 7, 8, 7, 8,
			8, 8, 8, 8, 7, 8, 7, 8, 8, 8, 8, 8
		}
	};

__device__ unsigned char getSPCX(CtxReg c, unsigned char i, unsigned char subband)
{
	return SPCXLUT[subband][(c >> (3 * i)) & 0x1FF];
}

	/* sign context in the following format
		index:
			first (MSB) bit V0 significance (1 significant, 0 insignificant)
			second bit V0 sign (0 positive, 1 negative)

			next 2 bits same for H0
			next 2 bits same for H1
			next 2 bits same for V1
			
		value:
			the response contains two pieces of information
			1. context label on the 4 least significant bits
			2. XORbit on the 5-th bit from the end (5-th least significant bit)
	*/

	__constant__ unsigned char signcxlut[256] = {
		 9,  9, 10, 26,  9,  9, 10, 26, 12, 12, 13, 11, 28, 28, 27, 29,  9,  9, 10, 26,
		 9,  9, 10, 26, 12, 12, 13, 11, 28, 28, 27, 29, 12, 12, 13, 11, 12, 12, 13, 11,
		12, 12, 13, 11,  9,  9, 10, 26, 28, 28, 27, 29, 28, 28, 27, 29,  9,  9, 10, 26,
		28, 28, 27, 29,  9,  9, 10, 26,  9,  9, 10, 26, 12, 12, 13, 11, 28, 28, 27, 29,
		 9,  9, 10, 26,  9,  9, 10, 26, 12, 12, 13, 11, 28, 28, 27, 29, 12, 12, 13, 11,
		12, 12, 13, 11, 12, 12, 13, 11,  9,  9, 10, 26, 28, 28, 27, 29, 28, 28, 27, 29,
		 9,  9, 10, 26, 28, 28, 27, 29, 10, 10, 10,  9, 10, 10, 10,  9, 13, 13, 13, 12,
		27, 27, 27, 28, 10, 10, 10,  9, 10, 10, 10,  9, 13, 13, 13, 12, 27, 27, 27, 28,
		13, 13, 13, 12, 13, 13, 13, 12, 13, 13, 13, 12, 10, 10, 10,  9, 27, 27, 27, 28,
		27, 27, 27, 28, 10, 10, 10,  9, 27, 27, 27, 28, 26, 26,  9, 26, 26, 26,  9, 26,
		11, 11, 12, 11, 29, 29, 28, 29, 26, 26,  9, 26, 26, 26,  9, 26, 11, 11, 12, 11,
		29, 29, 28, 29, 11, 11, 12, 11, 11, 11, 12, 11, 11, 11, 12, 11, 26, 26,  9, 26,
		29, 29, 28, 29, 29, 29, 28, 29, 26, 26,  9, 26, 29, 29, 28, 29
	};

__device__ unsigned char getSICX(CtxReg sig, CtxReg sign, unsigned char i)
{
	return signcxlut[
			((sig >> (i * 3)) & 0xAA) |
			(((sign >> (i * 3)) & 0xAA) >> 1)
		];
}

__device__ unsigned char getMRCX(CtxReg sig, CoefficientState local, unsigned char i)
{
	if((local >> (12 + 3 * i)) & 1)
		return 16;
	else if(((sig >> (3 * i)) & 0x1EF) == 0)
		return 14;
	else
		return 15;
}

class RLDecodeFunctor {
public:
	__device__ char operator()(CtxWindow &window, MQDecoder &dec)
	{
		char rest = 0;

		if(mqDecode(dec, CX_RUN) == 0)
		{
			rest = -2;
		}
		else
		{
			rest = mqDecode(dec, CX_UNI) & 1;
			rest <<= 1;
			rest |= mqDecode(dec, CX_UNI) & 1;

			window.c |= 1 << (3 * rest);
		}

		return rest;
	}
};

class SigDecodeFunctor {
public:
	__device__ void operator()(CtxWindow &window, CtxReg sig, MQDecoder &dec, int stripId, int subband)
	{
		window.c |= mqDecode(dec, getSPCX(sig, stripId, subband)) << (3 * stripId);
	}
};


class SignDecodeFunctor
{
public:
	__device__ void operator()(CtxWindow &window, CtxReg sig, MQDecoder &dec, int stripId)
	{
		unsigned char cx = getSICX(sig, buildCtxReg(window, 13), stripId);

		window.c |= (mqDecode(dec, cx & 0xF) ^ ((cx >> 4) & 1) & 1) << (13 + 3 * stripId);
	}	
};


class MagRefDecodeFunctor {
public:
	__device__ void operator()(MQDecoder &dec, CtxWindow &window, int stripId)
	{
		window.c |= (mqDecode(dec, getMRCX(buildCtxReg(window, 1), window.c, stripId)) << (3 * stripId));
	}
};

template <class MagRefCodingFunctor, typename MQCoderStateType>
class MagRefPassFunctor {
public:
__device__ void operator()(const CodeBlockAdditionalInfo &info, CtxWindow &window, MQCoderStateType &mq, float *sum_dist, unsigned char bitplane)
{
	for(int i = 0; i < 4; i++)
	{
		if(//csSignificant(st) && !csAlreadyCoded(st) && not out of bounds
			((window.c >> (3 * i)) & 0x4006) == 0x2)
		{
			*sum_dist -= (float)((1<<bitplane)*(1<<bitplane));
//			if(blockIdx.x * blockDim.x + threadIdx.x == 0)
//			printf("mgr:%f tid:%d\n", *sum_dist, blockIdx.x * blockDim.x + threadIdx.x);
			MagRefCodingFunctor()(mq, window, i);
			SetNthBit(window.c, 3 * i + 12);
		}
	}
}
};

__device__ void initDecodingCoeffs(const CodeBlockAdditionalInfo &info, CoefficientState *coeffs)
{
	for(int i = 0; i < info.width; i++)
		for(int j = 0; j < info.stripeNo; j++)
		{
			CoefficientState st = 0;

			for(int k = 0; k < 4; k++)
				if(4 * j + k < info.height)
					info.coefficients[(4 * j + k) * info.nominalWidth + i] = 0;
				else
					st |= (1 << (14 + 3 * k));

			coeffs[j * info.width + i] = st;
		}
}

__device__ void uploadSigns(const CodeBlockAdditionalInfo &info, CoefficientState *coeffs)
{
	unsigned char signOffset = sizeof(int) * 8 - 1;

	for(int i = 0; i < info.width; i++)
		for(int j = 0; j < info.stripeNo; j++)
		{
			CoefficientState st = coeffs[j * info.width + i];

			for(int k = 0; k < 4; k++)
				if(((st >> (14 + 3 * k)) & 1) == 0)
					info.coefficients[(4 * j + k) * info.nominalWidth + i] |= (((st >> (13 + 3 * k)) & 1) << signOffset);

			coeffs[j * info.width + i] = st;
		}
}

__device__ void fillMags(const CodeBlockAdditionalInfo &info, CoefficientState *coeffs, int bitplane)
{
	for(int i = 0; i < info.width; i++)
		for(int j = 0; j < info.stripeNo; j++)
		{
			CoefficientState st = coeffs[j * info.width + i];

			// clear magnitudes and already coded flags
			st &= ~(TRIMASK | (TRIMASK << 2));
			//st |= ((st & (TRIMASK << 1)) << 11);

			for(int k = 0; k < 4; k++)
				if(((st >> (14 + 3 * k)) & 1) == 0)
					st |= ((info.coefficients[(4 * j + k) * info.nominalWidth + i] >> bitplane) & 1) << (3 * k);

			coeffs[j * info.width + i] = st;
		}
}

__device__ void uploadMags(const CodeBlockAdditionalInfo &info, CoefficientState *coeffs, int bitplane)
{
	for(int i = 0; i < info.width; i++)
		for(int j = 0; j < info.stripeNo; j++)
		{
			CoefficientState st = coeffs[j * info.width + i];

			for(int k = 0; k < 4; k++)
				if(((st >> (14 + 3 * k)) & 1) == 0)
					info.coefficients[(4 * j + k) * info.nominalWidth + i] |= (((st >> (3 * k)) & 1) << bitplane);

			// clear magnitudes and already coded flags
			st &= ~(TRIMASK | (TRIMASK << 2));

			coeffs[j * info.width + i] = st;
		}
}

__device__ void clearWindow(CtxWindow &w)
{
	w.bl = 0;
	w.b = 0;
	w.br = 0;

	w.l = 0;
	w.c = 0;
	w.r = 0;

	w.tl = 0;
	w.t = 0;
	w.tr = 0;
}

template <class PassFunctor, typename MQCoderStateType>
__device__ void BITPLANE_WINDOW_SCAN(CodeBlockAdditionalInfo &info, CoefficientState *coeffs, MQCoderStateType &enc, float *sum_dist, unsigned char bitplane) {
	CtxWindow window;

	window.pos = -1;

	for(int j = 0; j < info.stripeNo; j++)
	{
		clearWindow(window);
		down(info, window, coeffs);
		shift(window);
		down(info, window, coeffs);
	
		PassFunctor()(info, window, enc, sum_dist, bitplane);

		for(int k = 0; k < info.width - 2; k++)
		{
			shift(window);
			down(info, window, coeffs);
			PassFunctor()(info, window, enc, sum_dist, bitplane);
			up(window, coeffs);
		}

		shift(window);
		PassFunctor()(info, window, enc, sum_dist, bitplane);
		up(window, coeffs);
		shift(window);
		up(window, coeffs);

		window.pos--;
	}
}

__device__ void decode(CoefficientState *coeffs, CodeBlockAdditionalInfo &info, byte *in)
{
	MQDecoder mqdec;
	mqInitDec(mqdec, in, info.length);

	float sum_dist = 0.0f;

	if(info.significantBits > 0)
	{
		mqResetDec(mqdec);

		initDecodingCoeffs(info, coeffs);

#ifdef CUDA

		BITPLANE_WINDOW_SCAN
		<CleanUpPassFunctor<RLDecodeFunctor, SigDecodeFunctor, SignDecodeFunctor, MQDecoder>, MQDecoder>
			(info, coeffs, mqdec, &sum_dist, 0);

		uploadMags(info, coeffs, 30 - info.magbits + info.significantBits);

		for(unsigned char i = 1; i < info.significantBits; i++)
		{
			BITPLANE_WINDOW_SCAN
			<SigPropPassFunctor<SigDecodeFunctor, SignDecodeFunctor, MQDecoder>, MQDecoder>
				(info, coeffs, mqdec, &sum_dist, 0);

			BITPLANE_WINDOW_SCAN
			<MagRefPassFunctor<MagRefDecodeFunctor, MQDecoder>, MQDecoder>
				(info, coeffs, mqdec, &sum_dist, 0);

			BITPLANE_WINDOW_SCAN
			<CleanUpPassFunctor<RLDecodeFunctor, SigDecodeFunctor, SignDecodeFunctor, MQDecoder>, MQDecoder>
				(info, coeffs, mqdec, &sum_dist, 0);

			uploadMags(info, coeffs, 30 - info.magbits - i + info.significantBits);
		}

#endif
		uploadSigns(info, coeffs);
	}
	else
	{
		for(int i = 0; i < info.height; i++)
			for(int j = 0; j < info.width; j++)
				info.coefficients[i * info.nominalWidth + j] = 0;
	}
}


__global__ void g_decode(CoefficientState *coeffBuffers, byte *inbuf, int maxThreadBufforLength, CodeBlockAdditionalInfo *infos, int codeBlocks)
{
	int threadId = 0;
#ifdef CUDA
	threadId = blockIdx.x * blockDim.x + threadIdx.x;
#endif

	if(threadId >= codeBlocks)
		return;

	CodeBlockAdditionalInfo info = infos[threadId];

	decode(coeffBuffers + info.magconOffset, info, inbuf + threadId * maxThreadBufforLength);
}


#ifdef CUDA
void launch_decode(dim3 gridDim, dim3 blockDim, CoefficientState *coeffBuffors, byte *inbuf, int maxThreadBufforLength, CodeBlockAdditionalInfo *infos, int codeBlocks)
{
	g_decode<<<gridDim, blockDim>>>(coeffBuffors, inbuf, maxThreadBufforLength, infos, codeBlocks);
}
#endif

}
