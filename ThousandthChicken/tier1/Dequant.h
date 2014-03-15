/*
 * Dequant.h
 *
 *  Created on: Mar 8, 2014
 *      Author: aaron
 */

#ifndef DEQUANT_H_
#define DEQUANT_H_

#include "../types/image_types.h"
#include "../print_info/print_info.h"

#define BLOCKSIZEX 16
#define BLOCKSIZEY 16
#define COMPUTED_ELEMS_BY_THREAD 4

class Dequant {
public:
	void dequantize_tile(type_tile *tile);

private:
	type_subband* dequantization(type_subband *sb);
	int get_exp_subband_gain(int orient);
};

#endif /* DEQUANT_H_ */
