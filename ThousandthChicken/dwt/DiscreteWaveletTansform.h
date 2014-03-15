/*
 * DiscreteWaveletTansform.h
 *
 *  Created on: Mar 8, 2014
 *      Author: aaron
 */

#ifndef DISCRETEWAVELETTANSFORM_H_
#define DISCRETEWAVELETTANSFORM_H_

#include "../types/image_types.h"

class DiscreteWaveletTansform {
public:
	 void iwt(type_tile *tile);

private:
	 type_data *iwt_2d(short filter, type_tile_comp *tile_comp);

};

#endif /* DISCRETEWAVELETTANSFORM_H_ */
