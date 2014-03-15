#ifndef GPU_MQ_CODER_CUH_
#define GPU_MQ_CODER_CUH_

#define __global__
#define __device__
#define __shared__
#define __constant__


__device__ __constant__ const unsigned short Qe[] = {
	0x5601,	0x3401,	0x1801,	0x0AC1, 0x0521, 0x0221, 0x5601, 0x5401, 0x4801, 0x3801,
	0x3001, 0x2401, 0x1C01, 0x1601, 0x5601, 0x5401, 0x5101, 0x4801, 0x3801, 0x3401,
	0x3001, 0x2801, 0x2401, 0x2201, 0x1C01, 0x1801, 0x1601, 0x1401, 0x1201, 0x1101,
	0x0AC1, 0x09C1, 0x08A1, 0x0521, 0x0441, 0x02A1, 0x0221, 0x0141, 0x0111, 0x0085,
	0x0049, 0x0025, 0x0015, 0x0009, 0x0005, 0x0001, 0x5601
};

__device__ __constant__ const unsigned char NMPS[] = {
	 1,  2,  3,  4,  5, 38,  7,  8,  9, 10, 11, 12, 13, 29, 15, 16, 17, 18, 19, 20,
	21, 22,	23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
	41, 42,	43, 44, 45, 45, 46
};

__device__ __constant__ const unsigned char NLPS[] = {
	 1,  6,  9, 12, 29, 33,  6, 14, 14, 14, 17, 18, 20, 21, 14, 14, 15, 16, 17, 18,
	19, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
	38, 39, 40, 41, 42, 43, 46
};

__device__ __constant__ const unsigned char SWITCH[] = {
	 1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,
	 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
	 0,  0,  0,  0,  0,  0,  0
};


typedef unsigned char byte;

#define CX_RUN 18
#define CX_UNI 17

struct MQEncoder
{
	short L;

	unsigned short A;
	unsigned int C;
	byte CT;
	byte T;

	byte *outbuf;

	int CXMPS;
	unsigned char CX;

	unsigned int Ib0;
	unsigned int Ib1;
	unsigned int Ib2;
	unsigned int Ib3;
	unsigned int Ib4;
	unsigned int Ib5;
};

struct MQDecoder : public MQEncoder
{
	byte NT;
	int Lmax;
};

__device__ unsigned char getI(MQEncoder &enc, int id)
{
	unsigned char out = 0;

	out |= ((enc.Ib0 >> id) & 1);
	out |= ((enc.Ib1 >> id) & 1) << 1;
	out |= ((enc.Ib2 >> id) & 1) << 2;
	out |= ((enc.Ib3 >> id) & 1) << 3;
	out |= ((enc.Ib4 >> id) & 1) << 4;
	out |= ((enc.Ib5 >> id) & 1) << 5;

	return out;
}

__device__ void setI(MQEncoder &enc, int id, unsigned char value)
{
	unsigned int mask = ~(1 << id);

	enc.Ib0 = (enc.Ib0 & mask) | (((value) & 1) << id);
	enc.Ib1 = (enc.Ib1 & mask) | (((value >> 1) & 1) << id);
	enc.Ib2 = (enc.Ib2 & mask) | (((value >> 2) & 1) << id);
	enc.Ib3 = (enc.Ib3 & mask) | (((value >> 3) & 1) << id);
	enc.Ib4 = (enc.Ib4 & mask) | (((value >> 4) & 1) << id);
	enc.Ib5 = (enc.Ib5 & mask) | (((value >> 5) & 1) << id);
}

__device__ void SwitchNthBit(int &reg, int n)
{
	reg = (reg ^ (1 << n));
}

__device__ short GetNthBit(int &reg, int n)
{
	return (reg >> n) & 1;
}

__device__ void mqResetDec(MQDecoder &decoder)
{
	decoder.Ib0 = 0;
	decoder.Ib1 = 0;
	decoder.Ib2 = 0;
	decoder.Ib3 = 0;
	decoder.Ib4 = 0;
	decoder.Ib5 = 0;

	setI(decoder, CX_UNI, 46);
	setI(decoder, CX_RUN, 3);
	setI(decoder, 0, 4);

	decoder.CXMPS = 0;
}

__device__ void bytein(MQDecoder &decoder)
{
	decoder.CT = 8;
	if(decoder.L == decoder.Lmax - 1 || (decoder.T == (unsigned char) 0xFF && decoder.NT > (unsigned char) 0x8F))
		decoder.C += 0xFF00;
	else
	{
		if(decoder.T == (unsigned char) 0xFF)
			decoder.CT = 7;

		decoder.T = decoder.NT;
		decoder.NT = decoder.outbuf[decoder.L + 1];
		decoder.L++;
		decoder.C += decoder.T << (16 - decoder.CT);
	}
}

__device__ void renormd(MQDecoder &decoder)
{
	do
	{
		if(decoder.CT == 0)
			bytein(decoder);

		decoder.A <<= 1;
		decoder.C <<= 1;
		decoder.CT -= 1;
	}
	while((decoder.A & 0x8000) == 0);
}

__device__ int lps_exchange(MQDecoder &decoder)
{
	unsigned int p = Qe[getI(decoder, decoder.CX)];
	int D;

	if(decoder.A < p)
	{
		decoder.A = p;
		D = GetNthBit(decoder.CXMPS, decoder.CX);
		setI(decoder, decoder.CX, NMPS[getI(decoder, decoder.CX)]);
	}
	else
	{
		decoder.A = p;
		D = 1 - GetNthBit(decoder.CXMPS, decoder.CX);

		if(SWITCH[getI(decoder, decoder.CX)])
			SwitchNthBit(decoder.CXMPS, decoder.CX);

		setI(decoder, decoder.CX, NLPS[getI(decoder, decoder.CX)]);
	}

	return D;
}

__device__ int mps_exchange(MQDecoder &decoder)
{
	unsigned int p = Qe[getI(decoder, decoder.CX)];
	int D;

	if(decoder.A < p)
	{
		D = 1 - GetNthBit(decoder.CXMPS, decoder.CX);

		if(SWITCH[getI(decoder, decoder.CX)])
			SwitchNthBit(decoder.CXMPS, decoder.CX);
		
		setI(decoder, decoder.CX, NLPS[getI(decoder, decoder.CX)]);
	}
	else
	{
		D = GetNthBit(decoder.CXMPS, decoder.CX);
		setI(decoder, decoder.CX, NMPS[getI(decoder, decoder.CX)]);
	}

	return D;
}

__device__ void mqInitDec(MQDecoder &decoder, byte *inbuf, int codeLength)
{

	decoder.outbuf = inbuf;

	decoder.L = -1;
	decoder.Lmax = codeLength;
	decoder.T = 0;
	decoder.NT = 0;

	bytein(decoder);
	bytein(decoder);

	decoder.C = ((unsigned char) decoder.T) << 16;

	bytein(decoder);

	decoder.C <<= 7;
	decoder.CT -= 7;
	decoder.A = 0x8000;
}

__device__ int mqDecode(MQDecoder &decoder, int context)
{
	decoder.CX = context;

	unsigned int p = Qe[getI(decoder, decoder.CX)];
	int out;
	decoder.A -= p;

	if((decoder.C >> 16) < p)
	{
		out = lps_exchange(decoder);
		renormd(decoder);
	}
	else
	{
		// decrement 16 most significant bits of C register by p
		decoder.C = (decoder.C & 0x0000FFFF) | (((decoder.C >> 16) - p) << 16);

		if((decoder.A & 0x8000) == 0)
		{
			out = mps_exchange(decoder);
			renormd(decoder);
		}
		else
		{
			out = GetNthBit(decoder.CXMPS, decoder.CX);
		}
	}

	#ifdef DEBUG_MQ
	/* debug purposes */ 
	Cstates[l++] = 1;
	Cstates[l++] = out;
	Cstates[l++] = context;
	/* */
	#endif

	return out;
}

#endif /* GPU_MQ_CODER_CUH_ */
