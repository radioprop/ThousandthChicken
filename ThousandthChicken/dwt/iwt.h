#ifndef IWT_H_
#define IWT_H_

#define __global__
#define __device__
#define __shared__


#define SHIFT 4
#define ADD_VALUES 7

#define MEMSIZE 40
#define PATCHX 32
#define PATCHY 32
#define PATCHX_DIV_2 16
#define PATCHY_DIV_2 16
#define BLOCKSIZEX 16
#define BLOCKSIZEY 16

#define OFFSET_97 4
#define OFFSET_53 2
#define IOFFSET_97 3
#define IOFFSET_53 2
#define OFFSETX_DIV_2 2
#define OFFSETY_DIV_2 2
#define FIRST_BLOCK 0

#ifdef CUDA
extern __global__
void iwt97(const float *idata, float *odata, const int2 img_size, const int2 step);

extern __global__
void iwt53(const float *idata, float *odata, const int2 img_size, const int2 step);
#endif


#endif /* IWT_H_ */
