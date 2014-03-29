
//mq coder //////////////////////////////////////////////////

#include "tier1/coeff_coder/gpu_coeff_coder2.h"

typedef struct MQEncoder
{
	short L;

	unsigned short A;
	unsigned int C;
	unsigned char CT;
	unsigned char T;

	__global unsigned char *outbuf;

	int CXMPS;
	unsigned char CX;

	unsigned int Ib0;
	unsigned int Ib1;
	unsigned int Ib2;
	unsigned int Ib3;
	unsigned int Ib4;
	unsigned int Ib5;
} MQEncoder;

typedef struct MQDecoder
{
	MQEncoder encoder;
	unsigned char NT;
	int Lmax;
} MQDecoder;


__constant const unsigned short Qe[] = {
	0x5601,	0x3401,	0x1801,	0x0AC1, 0x0521, 0x0221, 0x5601, 0x5401, 0x4801, 0x3801,
	0x3001, 0x2401, 0x1C01, 0x1601, 0x5601, 0x5401, 0x5101, 0x4801, 0x3801, 0x3401,
	0x3001, 0x2801, 0x2401, 0x2201, 0x1C01, 0x1801, 0x1601, 0x1401, 0x1201, 0x1101,
	0x0AC1, 0x09C1, 0x08A1, 0x0521, 0x0441, 0x02A1, 0x0221, 0x0141, 0x0111, 0x0085,
	0x0049, 0x0025, 0x0015, 0x0009, 0x0005, 0x0001, 0x5601
};

__constant const unsigned char NMPS[] = {
	 1,  2,  3,  4,  5, 38,  7,  8,  9, 10, 11, 12, 13, 29, 15, 16, 17, 18, 19, 20,
	21, 22,	23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
	41, 42,	43, 44, 45, 45, 46
};

__constant const unsigned char NLPS[] = {
	 1,  6,  9, 12, 29, 33,  6, 14, 14, 14, 17, 18, 20, 21, 14, 14, 15, 16, 17, 18,
	19, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
	38, 39, 40, 41, 42, 43, 46
};

__constant const unsigned char SWITCH[] = {
	 1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,
	 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
	 0,  0,  0,  0,  0,  0,  0
};




#define CX_RUN 18
#define CX_UNI 17


 unsigned char getI(MQEncoder* encoder, int id)
{
	unsigned char out = 0;

	out |= ((encoder->Ib0 >> id) & 1);
	out |= ((encoder->Ib1 >> id) & 1) << 1;
	out |= ((encoder->Ib2 >> id) & 1) << 2;
	out |= ((encoder->Ib3 >> id) & 1) << 3;
	out |= ((encoder->Ib4 >> id) & 1) << 4;
	out |= ((encoder->Ib5 >> id) & 1) << 5;

	return out;
}

 void setI(MQEncoder* encoder, int id, unsigned char value)
{
	unsigned int mask = ~(1 << id);

	encoder->Ib0 = (encoder->Ib0 & mask) | (((value) & 1) << id);
	encoder->Ib1 = (encoder->Ib1 & mask) | (((value >> 1) & 1) << id);
	encoder->Ib2 = (encoder->Ib2 & mask) | (((value >> 2) & 1) << id);
	encoder->Ib3 = (encoder->Ib3 & mask) | (((value >> 3) & 1) << id);
	encoder->Ib4 = (encoder->Ib4 & mask) | (((value >> 4) & 1) << id);
	encoder->Ib5 = (encoder->Ib5 & mask) | (((value >> 5) & 1) << id);
}

 void SwitchNthBit(int reg, int n)
{
	reg = (reg ^ (1 << n));
}

 short GetNthBit(int reg, int n)
{
	return (reg >> n) & 1;
}

 void mqResetDec(MQDecoder* decoder)
{
	decoder->encoder.Ib0 = 0;
	decoder->encoder.Ib1 = 0;
	decoder->encoder.Ib2 = 0;
	decoder->encoder.Ib3 = 0;
	decoder->encoder.Ib4 = 0;
	decoder->encoder.Ib5 = 0;

	setI(&decoder->encoder, CX_UNI, 46);
	setI(&decoder->encoder, CX_RUN, 3);
	setI(&decoder->encoder, 0, 4);

	decoder->encoder.CXMPS = 0;
}

 void bytein(MQDecoder* decoder)
{
	decoder->encoder.CT = 8;
	if(decoder->encoder.L == decoder->Lmax - 1 || (decoder->encoder.T == (unsigned char) 0xFF && decoder->NT > (unsigned char) 0x8F))
		decoder->encoder.C += 0xFF00;
	else
	{
		if(decoder->encoder.T == (unsigned char) 0xFF)
			decoder->encoder.CT = 7;

		decoder->encoder.T = decoder->NT;
		decoder->NT = decoder->encoder.outbuf[decoder->encoder.L + 1];
		decoder->encoder.L++;
		decoder->encoder.C += decoder->encoder.T << (16 - decoder->encoder.CT);
	}
}

 void renormd(MQDecoder* decoder)
{
	do
	{
		if(decoder->encoder.CT == 0)
			bytein(decoder);

		decoder->encoder.A <<= 1;
		decoder->encoder.C <<= 1;
		decoder->encoder.CT -= 1;
	}
	while((decoder->encoder.A & 0x8000) == 0);
}

 int lps_exchange(MQDecoder* decoder)
{
	int D;
	unsigned int p = Qe[getI(&decoder->encoder, decoder->encoder.CX)];


	if(decoder->encoder.A < p)
	{
		decoder->encoder.A = p;
		D = GetNthBit(decoder->encoder.CXMPS, decoder->encoder.CX);
		setI(&decoder->encoder, decoder->encoder.CX, NMPS[getI(&decoder->encoder, decoder->encoder.CX)]);
	}
	else
	{
		decoder->encoder.A = p;
		D = 1 - GetNthBit(decoder->encoder.CXMPS, decoder->encoder.CX);

		if(SWITCH[getI(&decoder->encoder, decoder->encoder.CX)])
			SwitchNthBit(decoder->encoder.CXMPS, decoder->encoder.CX);

		setI(&decoder->encoder, decoder->encoder.CX, NLPS[getI(&decoder->encoder, decoder->encoder.CX)]);
	}

	return D;
}

 int mps_exchange(MQDecoder* decoder)
{
	int D;
	unsigned int p = Qe[getI(&decoder->encoder, decoder->encoder.CX)];


	if(decoder->encoder.A < p)
	{
		D = 1 - GetNthBit(decoder->encoder.CXMPS, decoder->encoder.CX);

		if(SWITCH[getI(&decoder->encoder, decoder->encoder.CX)])
			SwitchNthBit(decoder->encoder.CXMPS, decoder->encoder.CX);
		
		setI(&decoder->encoder, decoder->encoder.CX, NLPS[getI(&decoder->encoder, decoder->encoder.CX)]);
	}
	else
	{
		D = GetNthBit(decoder->encoder.CXMPS, decoder->encoder.CX);
		setI(&decoder->encoder, decoder->encoder.CX, NMPS[getI(&decoder->encoder, decoder->encoder.CX)]);
	}

	return D;
}

 void mqInitDec(MQDecoder* decoder, __global unsigned char *inbuf, int codeLength)
{
	decoder->encoder.outbuf = inbuf;

	decoder->encoder.L = -1;
	decoder->Lmax = codeLength;
	decoder->encoder.T = 0;
	decoder->NT = 0;

	bytein(decoder);
	bytein(decoder);

	decoder->encoder.C = ((unsigned char) decoder->encoder.T) << 16;

	bytein(decoder);

	decoder->encoder.C <<= 7;
	decoder->encoder.CT -= 7;
	decoder->encoder.A = 0x8000;
}

 int mqDecode(MQDecoder* decoder, int context)
{
	decoder->encoder.CX = context;

	unsigned int p = Qe[getI(&decoder->encoder, decoder->encoder.CX)];
	int out;
	decoder->encoder.A -= p;

	if((decoder->encoder.C >> 16) < p)
	{
		out = lps_exchange(decoder);
		renormd(decoder);
	}
	else
	{
		// decrement 16 most significant bits of C register by p
		decoder->encoder.C = (decoder->encoder.C & 0x0000FFFF) | (((decoder->encoder.C >> 16) - p) << 16);

		if((decoder->encoder.A & 0x8000) == 0)
		{
			out = mps_exchange(decoder);
			renormd(decoder);
		}
		else
		{
			out = GetNthBit(decoder->encoder.CXMPS, decoder->encoder.CX);
		}
	}
	return out;
}
////////////////////////////////////////////////////////////////////////////////

 
void SetMaskedBits(unsigned int reg, unsigned int mask, unsigned int bits)
{
	reg = (reg & ~mask) | (bits & mask);
}

void SetNthBit(unsigned int reg, unsigned int n)
{
	SetMaskedBits(reg, 1 << n, 1 << n);
}

void ResetNthBit(unsigned int reg, unsigned int n)
{
	SetMaskedBits(reg, 1 << n, 0);
}

typedef struct
{
	unsigned int tl;
	unsigned int  t;
	unsigned int  tr;
	
	unsigned int  l;
	unsigned int  c;
	unsigned int  r;
	
	unsigned int  bl;
	unsigned int  b;
	unsigned int  br;

	short pos;
} CtxWindow;

 void debug_print(float *val, int tid)
{
//	if(tid == 3)
//		printf("dist:%f\n", *val);
}


 void down(CodeBlockAdditionalInfo info, CtxWindow *window, __global  unsigned int  *coeffs)
{
	window->tr = coeffs[window->pos + 1 - info.width];
	window->r = coeffs[window->pos + 1];
	window->br = coeffs[window->pos + 1 + info.width];
}

 void up(CtxWindow *window, __global unsigned int  *coeffs)
{
	coeffs[window->pos - 1] = window->l;
}

 void shift(CtxWindow *window)
{
	window->tl = window->t; window->t = window->tr; window->tr = 0; // top layer
	window->l = window->c; window->c = window->r; window->r = 0; // middle layer
	window->bl = window->b; window->b = window->br; window->br = 0; // bottom layer
	window->pos += 1;
}

typedef int CtxReg;

#define TRIMASK 0x249 //((1 << 0) | (1 << 3) | (1 << 6) | (1 << 9))

 CtxReg buildCtxReg(CtxWindow *window, unsigned char bitoffset)
{
	CtxReg reg = 0;

	reg |= ((window->tl >> (bitoffset + 9)) & 1) << 0;
	reg |= ((window->t >> (bitoffset + 9)) & 1) << 1;
	reg |= ((window->tr >> (bitoffset + 9)) & 1) << 2;
	reg |= ((window->l >> (bitoffset)) & TRIMASK) << 3;
	reg |= ((window->c >> (bitoffset)) & TRIMASK) << 4;
	reg |= ((window->r >> (bitoffset)) & TRIMASK) << 5;
	reg |= ((window->bl >> (bitoffset)) & 1) << 15;
	reg |= ((window->b >> (bitoffset)) & 1) << 16;
	reg |= ((window->br >> (bitoffset)) & 1) << 17;
	
	return reg;
}


	__constant unsigned char SPCXLUT[3][512] = {
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

 unsigned char getSPCX(CtxReg c, unsigned char i, unsigned char subband)
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

	__constant unsigned char signcxlut[256] = {
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

 unsigned char getSICX(CtxReg sig, CtxReg sign, unsigned char i)
{
	return signcxlut[
			((sig >> (i * 3)) & 0xAA) |
			(((sign >> (i * 3)) & 0xAA) >> 1)
		];
}

 unsigned char getMRCX(CtxReg sig, unsigned int  localVal, unsigned char i)
{
	if((localVal >> (12 + 3 * i)) & 1)
		return 16;
	else if(((sig >> (3 * i)) & 0x1EF) == 0)
		return 14;
	else
		return 15;
}

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

__constant float distWeights[2][4][4] = {
{//Lossless
//		LH,      HL,      HH,     LLend
	{0.1000f, 0.1000f, 0.0500f, 1.0000f},  //level 0 = biggest subbands (unimportant)
	{0.2000f, 0.2000f, 0.1000f, 1.0000f},  //      1
	{0.4000f, 0.4000f, 0.2000f, 1.0000f},  //      2
	{0.8000f, 0.8000f, 0.4000f, 1.0000f}   //      3 = smallest, contains LL
}, {//Lossy
/*	{ 0.0010f, 0.0010f, 0.0005f, 1.0000f},
	{ 0.1000f, 0.1000f, 0.0250f, 1.0000f},
	{ 0.3000f, 0.3000f, 0.0800f, 1.0000f},
	{ 0.8000f, 0.8000f, 0.4000f, 1.0000f}*/
	{0.0100f, 0.0100f, 0.0050f, 1.0000f},
	{0.2000f, 0.2000f, 0.1000f, 1.0000f},
	{0.4000f, 0.4000f, 0.2000f, 1.0000f},
	{0.8000f, 0.8000f, 0.4000f, 1.0000f}
} };

 float getDISW(CodeBlockAdditionalInfo info)
{
	return distWeights[info.compType][MIN(info.dwtLevel, 3)][info.subband] * info.stepSize * info.stepSize / ((float)(info.width * info.height));
}


 char RLDecodeFunctor(CtxWindow *window, MQDecoder* dec)
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

		window->c |= 1 << (3 * rest);
	}

	return rest;
}




 void SigDecodeFunctor(CtxWindow *window, CtxReg sig, MQDecoder* dec, int stripId, int subband)
{
	window->c |= mqDecode(dec, getSPCX(sig, stripId, subband)) << (3 * stripId);
}




 void SignDecodeFunctor(CtxWindow *window, CtxReg sig, MQDecoder* dec, int stripId)
{
	unsigned char cx = getSICX(sig, buildCtxReg(window, 13), stripId);

	window->c |= (mqDecode(dec, cx & 0xF) ^ ((cx >> 4) & 1) & 1) << (13 + 3 * stripId);
}	




 void CleanUpPassFunctor(const CodeBlockAdditionalInfo info, CtxWindow *window, MQDecoder* mq, float *sum_dist, unsigned char bitplane)
{
	char rest;

	CtxReg sig = buildCtxReg(window, 1); // significance context

	rest = -1;
	if((window->c & (TRIMASK << 14)) == 0 && sig == 0) // all contexts in stripe are equal to zero
	{
		rest = RLDecodeFunctor(window, mq);
		if(rest == -2)
			return;
	}

	for(int k = 0; k < 4; k++)
	{
		if(/*	((window->c >> ( 1 + 3 * k)) & 1) == 0 &&   // check if coefficient is non-significant (sigma)
			((window->c >> ( 2 + 3 * k)) & 1) == 0 &&   // check if coefficient hasn't been coded already (pi)
			((window->c >> (14 + 3 * k)) & 1) == 0)    // forbidden state indicating out of bounds (late sigma)*/
			((window->c >> (3 * k)) & 0x4006) == 0)
		{
			if(rest >= 0)
				rest--;
			else
				SigDecodeFunctor(window, sig, mq, k, info.subband);
			
			if((window->c >> (3 * k)) & 1) // check if magnitude is 1
			{
				*sum_dist -= (float)((1<<bitplane)*(1<<bitplane));
//					debug_print(sum_dist, threadIdx.x);
//					if(blockIdx.x * blockDim.x + threadIdx.x == 0)
//					printf("clu:%f tid:%d\n", *sum_dist, blockIdx.x * blockDim.x + threadIdx.x);
				SetNthBit(window->c, 1 + 3 * k); // set k-th significant state
				sig = buildCtxReg(window, 1); // rebuild significance register

				SignDecodeFunctor(window, sig, mq, k);
			}
		}
	}
}



 void SigPropPassFunctor(const CodeBlockAdditionalInfo info, CtxWindow *window, MQDecoder* mq, float *sum_dist, unsigned char bitplane)
{
	CtxReg sig = buildCtxReg(window, 1); // build significance context register

	for(int i = 0; i < 4; i++)
	{
		// not significant with non-zero context
		if(/*	((window->c >> (1 + 3 * i)) & 1) == 0 &&
			((window->c >> (14 + 3 * i)) & 1) == 0 && // out of bounds
			getSPCX(sig, i, subband) > 0)*/
			(((window->c >> (3 * i)) & 0x4002) == 0) &&
			((sig >> (3 * i)) & 0x1EF) != 0)
		{
			SigDecodeFunctor(window, sig, mq, i, info.subband);

			// if magnitude bit is one
			if((window->c >> (3 * i)) & 1)
			{
				*sum_dist -= (float)((1<<bitplane)*(1<<bitplane));
//				debug_print(sum_dist, threadIdx.x);
//				if(blockIdx.x * blockDim.x + threadIdx.x == 0)
//				printf("sig:%f tid:%d\n", *sum_dist, blockIdx.x * blockDim.x + threadIdx.x);
				SetNthBit(window->c, 1 + (3 * i));
				sig = buildCtxReg(window, 1); // rebuild

				SignDecodeFunctor(window, sig, mq, i);
			}

			// set pi (already coded)
			SetNthBit(window->c, 2 + (3 * i));
		}
		else
			// unset pi (already coded)
			ResetNthBit(window->c, 2 + (3 * i));
	}
}




 void MagRefPassFunctor(const CodeBlockAdditionalInfo info, CtxWindow *window, MQDecoder* mq, float *sum_dist, unsigned char bitplane)
{
	for(int i = 0; i < 4; i++)
	{
		if(//csSignificant(st) && !csAlreadyCoded(st) && not out of bounds
			((window->c >> (3 * i)) & 0x4006) == 0x2)
		{
			*sum_dist -= (float)((1<<bitplane)*(1<<bitplane));
//			debug_print(sum_dist, threadIdx.x);
//			if(blockIdx.x * blockDim.x + threadIdx.x == 0)
//			printf("mgr:%f tid:%d\n", *sum_dist, blockIdx.x * blockDim.x + threadIdx.x);
			//MagRefCodingFunctor()(mq, window, i);
			window->c |= (mqDecode(mq, getMRCX(buildCtxReg(window, 1), window->c, i)) << (3 * i));
			SetNthBit(window->c, 3 * i + 12);
		}
	}
}

 void initDecodingCoeffs(const CodeBlockAdditionalInfo info, __global unsigned int  *coeffs,__global int* decodedCoefficients)
{
    int maxIndex =   sizeof(int) * info.nominalWidth * info.nominalHeight;
	for(int i = 0; i < info.width; i++)
		for(int j = 0; j < info.stripeNo; j++)
		{
			unsigned int  st = 0;

			for(int k = 0; k < 4; k++)
				if(4 * j + k < info.height)
				{
				   int index = (4 * j + k) * info.nominalWidth + i;
				   if (index < maxIndex)
					     decodedCoefficients[index] = 0;
				}
					
				else
				  st |= (1 << (14 + 3 * k));

			coeffs[j * info.width + i] = st;
		}
}

 void uploadSigns(const CodeBlockAdditionalInfo info, __global unsigned int  *coeffs, __global int* decodedCoefficients)
{
	unsigned char signOffset = sizeof(int) * 8 - 1;

	for(int i = 0; i < info.width; i++)
		for(int j = 0; j < info.stripeNo; j++)
		{
			unsigned int  st = coeffs[j * info.width + i];

			for(int k = 0; k < 4; k++)
				if(((st >> (14 + 3 * k)) & 1) == 0)
					decodedCoefficients[(4 * j + k) * info.nominalWidth + i] |= (((st >> (13 + 3 * k)) & 1) << signOffset);

			coeffs[j * info.width + i] = st;
		}
}

 void fillMags(const CodeBlockAdditionalInfo info, __global unsigned int  *coeffs, int bitplane, __global int* decodedCoefficients)
{
	for(int i = 0; i < info.width; i++)
		for(int j = 0; j < info.stripeNo; j++)
		{
			unsigned int  st = coeffs[j * info.width + i];

			// clear magnitudes and already coded flags
			st &= ~(TRIMASK | (TRIMASK << 2));
			//st |= ((st & (TRIMASK << 1)) << 11);

			for(int k = 0; k < 4; k++)
				if(((st >> (14 + 3 * k)) & 1) == 0)
					st |= ((decodedCoefficients[(4 * j + k) * info.nominalWidth + i] >> bitplane) & 1) << (3 * k);

			coeffs[j * info.width + i] = st;
		}
}

 void uploadMags(const CodeBlockAdditionalInfo info, __global unsigned int  *coeffs, int bitplane, __global int* decodedCoefficients)
{
	for(int i = 0; i < info.width; i++)
		for(int j = 0; j < info.stripeNo; j++)
		{
			unsigned int  st = coeffs[j * info.width + i];

			for(int k = 0; k < 4; k++)
				if(((st >> (14 + 3 * k)) & 1) == 0)
					decodedCoefficients[(4 * j + k) * info.nominalWidth + i] |= (((st >> (3 * k)) & 1) << bitplane);

			// clear magnitudes and already coded flags
			st &= ~(TRIMASK | (TRIMASK << 2));

			coeffs[j * info.width + i] = st;
		}
}

 void clearWindow(CtxWindow *w)
{
	w->bl = 0;
	w->b = 0;
	w->br = 0;

	w->l = 0;
	w->c = 0;
	w->r = 0;

	w->tl = 0;
	w->t = 0;
	w->tr = 0;
}

 
 void BITPLANE_WINDOW_SCAN_CLEAN( CodeBlockAdditionalInfo info, __global unsigned int  *coeffs, MQDecoder* enc, float *sum_dist, unsigned char bitplane) {
	
	 CtxWindow window;
 	 window.pos = -1;

	for(int j = 0; j < info.stripeNo; j++)
	{
		clearWindow(&window);
		down(info, &window, coeffs);
		shift(&window);
		down(info, &window, coeffs);
	
		CleanUpPassFunctor(info, &window, enc, sum_dist, bitplane);

		for(int k = 0; k < info.width - 2; k++)
		{
			shift(&window);
			down(info, &window, coeffs);
			CleanUpPassFunctor(info, &window, enc, sum_dist, bitplane);
			up(&window, coeffs);
		}

		shift(&window);
		CleanUpPassFunctor(info, &window, enc, sum_dist, bitplane);
		up(&window, coeffs);
		shift(&window);
		up(&window, coeffs);

		window.pos--;
	}
}

 
 void BITPLANE_WINDOW_SCAN_MAG( CodeBlockAdditionalInfo info, __global unsigned int  *coeffs, MQDecoder* enc, float *sum_dist, unsigned char bitplane) {
	
	 CtxWindow window;
 	 window.pos = -1;

	for(int j = 0; j < info.stripeNo; j++)
	{
		clearWindow(&window);
		down(info, &window, coeffs);
		shift(&window);
		down(info, &window, coeffs);
	
		MagRefPassFunctor(info, &window, enc, sum_dist, bitplane);

		for(int k = 0; k < info.width - 2; k++)
		{
			shift(&window);
			down(info, &window, coeffs);
			MagRefPassFunctor(info, &window, enc, sum_dist, bitplane);
			up(&window, coeffs);
		}

		shift(&window);
		MagRefPassFunctor(info, &window, enc, sum_dist, bitplane);
		up(&window, coeffs);
		shift(&window);
		up(&window, coeffs);

		window.pos--;
	}
}

 
 void BITPLANE_WINDOW_SCAN_SIG( CodeBlockAdditionalInfo info, __global unsigned int  *coeffs, MQDecoder* enc, float *sum_dist, unsigned char bitplane) {
	
	 CtxWindow window;
 	 window.pos = -1;

	for(int j = 0; j < info.stripeNo; j++)
	{
		clearWindow(&window);
		down(info, &window, coeffs);
		shift(&window);
		down(info, &window, coeffs);
	
		SigPropPassFunctor(info, &window, enc, sum_dist, bitplane);

		for(int k = 0; k < info.width - 2; k++)
		{
			shift(&window);
			down(info, &window, coeffs);
			SigPropPassFunctor(info, &window, enc, sum_dist, bitplane);
			up(&window, coeffs);
		}

		shift(&window);
		SigPropPassFunctor(info, &window, enc, sum_dist, bitplane);
		up(&window, coeffs);
		shift(&window);
		up(&window, coeffs);

		window.pos--;
	}
}

__kernel void g_decode(__global unsigned int *stBuffers, __global unsigned char *codestreamBuffer, 
                            int maxThreadBufferLength, __global CodeBlockAdditionalInfo *codeblockInfoArray, 
							  int codeBlocks,__global int* decodedCoefficientsBuffer)
{

	
	size_t idx = get_global_id(0);
	if(idx >= codeBlocks)
		return;

	CodeBlockAdditionalInfo codeblockInfo = codeblockInfoArray[idx];
	__global unsigned char *codestream = codestreamBuffer + idx * maxThreadBufferLength;
	__global unsigned int* st = stBuffers + codeblockInfo.magconOffset;
	__global int* decodedCoefficients = decodedCoefficientsBuffer + codeblockInfo.gpuCoefficientsOffset;

	MQDecoder mqdec;
	mqInitDec(&mqdec, codestream, codeblockInfo.length);
	float sum_dist = 0.0f;

	if(codeblockInfo.significantBits > 0)
	{
		mqResetDec(&mqdec);

		initDecodingCoeffs(codeblockInfo, st, decodedCoefficients);
		
		BITPLANE_WINDOW_SCAN_CLEAN(codeblockInfo, st, &mqdec, &sum_dist, 0);

		uploadMags(codeblockInfo, st, 30 - codeblockInfo.magbits + codeblockInfo.significantBits,decodedCoefficients);

		for(unsigned char i = 1; i < codeblockInfo.significantBits; i++)
		{
			BITPLANE_WINDOW_SCAN_SIG(codeblockInfo, st, &mqdec, &sum_dist, 0);

			BITPLANE_WINDOW_SCAN_MAG(codeblockInfo, st, &mqdec, &sum_dist, 0);

			BITPLANE_WINDOW_SCAN_CLEAN(codeblockInfo, st, &mqdec, &sum_dist, 0);

			uploadMags(codeblockInfo, st, 30 - codeblockInfo.magbits - i + codeblockInfo.significantBits,decodedCoefficients);
		}

		uploadSigns(codeblockInfo, st,decodedCoefficients);
		
	}

	
	
}






