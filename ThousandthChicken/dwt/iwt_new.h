#ifndef IWT_NEW_H_
#define IWT_NEW_H_

#define __global__
#define __device__
#define __shared__

#ifdef CUDA
extern __global__
void iwt53_new(const float *idata, float *odata, const int2 img_size, const int2 step);
#endif

#endif /* IWT_NEW_H_ */
