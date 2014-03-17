#ifndef IMAGE_MCT_H_
#define IMAGE_MCT_H_

#define MCT_8BIT_INT 0
#define MCT_16BIT_INT 1
#define MCT_32BIT_FLOAT 2
#define MCT_64BIT_DOUBLE 3
#define MCT_128BIT_DOUBLE 4 /* actually used only within wavelet context in ATK segment marker */

#define MCT_DECORRELATION_TRANSFORMATION 0
#define MCT_DEPENDENCY_TRANSFORMATION 1
#define MCT_DECORRELATION_OFFSET 2
#define MCT_DEPENDENCY_OFFSET 3

#define MCC_MATRIX_BASED 0
#define MCC_WAVELET_BASED_LOW 2
#define MCC_WAVELET_BASED_HIGH 3

typedef struct type_mct type_mct;
typedef struct type_mcc type_mcc;
typedef struct type_mcc_data type_mcc_data;
typedef struct type_mic type_mic;
typedef struct type_mic_data type_mic_data;
typedef struct type_atk type_atk;
typedef struct type_ads type_ads;
typedef struct type_multiple_component_transformations type_multiple_component_transformations;

/** Data gathering point for multiple component transformation as in 15444-2 Annex I */ 
struct type_multiple_component_transformations
{
	/** Transformation matrices */
	type_mct* mcts[4];
	/** Number of transformation matrices by type */
	unsigned char mcts_count[4];

	/** Multiple component collection segments */
	type_mcc* mccs;

	/** Count of component collection segments */
	unsigned char mccs_count;

	/** Multiple intermediate collection segments */
	type_mic* mics;

	/** Count of intermediate collection segments */
	unsigned char mics_count;

	/** Arbitrary decomposition styles */
	type_ads* adses;

	/** Count of ADS segments */
	unsigned char ads_count;

	/** Arbitrary transformation kernels */
	type_atk* atks;

	/** Count of ATK segemnts */
	unsigned char atk_count;
};

/** MCT as in 15444-2 Annex A.3.7 */
struct type_mct
{
	/** Matrix definition index */
	unsigned char index;

	/** Matrix type */
	unsigned char type;

	/** Matrix element data type */
	unsigned char element_type;

	/** Element count */
	unsigned int length;

	/** Data */
	unsigned char* data;
};

/** MCC as in 15444-2 Annex A.3.8 */
struct type_mcc {
	/** Index of marker segment */
	unsigned char index;

	/** Count of collections in segment */
	unsigned char count;

	/** Component collections */
	type_mcc_data* data;
};

/** Component Collection part of MCC segment as in 15444-2 Annex A.3.8 */
struct type_mcc_data {
	/** Decorrelation type */
	unsigned char type;

	/** Number of input components */
	unsigned short input_count;

	/** Are components number 8 or 16 bit? */
	unsigned char input_component_type;

	/** Input components identifiers */
	unsigned char* input_components;

	/** Number of output components */
	unsigned short output_count;

	/** Are components number 8 or 16 bit? */
	unsigned char output_component_type;

	/** Input components identifiers */
	unsigned char* output_components;

	/** Number of transform matrix to use in decorrelation process 
	 *		Used only with matrix based decorrelation!
	 */
	unsigned char decorrelation_transform_matrix;

	/** Number of transform offset matrix to use in decorrelation process 
	 *		Used only with matrix based decorrelation!
	 */
	unsigned char deccorelation_transform_offset;

	/** Index of ATK marker
	 * 	Used only with wavelet based decorrelation!
	 */
	unsigned char atk;

	/** Index of ADS marker
	 * 	Used only with wavelet based decorrelation!
	 */
	unsigned char ads;
};

/** MIC as in 15444-2 Annex A.3.9 */
struct type_mic {
	/** Index of marker segment */
	unsigned char index;

	/** Count of collections in segment */
	unsigned char count;

	/** Component collections */
	type_mic_data* data;
};

/** Component Intermediate Collection part of MIC segment as in 15444-2 Annex A.3.9 */
struct type_mic_data {
	/** Number of input components */
	unsigned short input_count;

	/** Are components number 8 or 16 bit? */
	unsigned char input_component_type;

	/** Input components identifiers */
	unsigned char* input_components;

	/** Number of output components */
	unsigned short output_count;

	/** Are components number 8 or 16 bit? */
	unsigned char output_component_type;

	/** Input components identifiers */
	unsigned char* output_components;

	/** Number of transform matrix to use in decorrelation process */ 
	unsigned char decorrelation_transform_matrix;

	/** Number of transform offset matrix to use in decorrelation process */ 
	unsigned char deccorelation_transform_offset;
};

struct type_ads {
	/** Index of marker segment */
	unsigned char index;
	
	/** Number of elements in the string defining the number of decomposition sub-levels */
	unsigned char IOads;

	/** String defining the number of decomposition sub-levels. */
	unsigned char* DOads;

	/** Number of elements in the string defining the arbitrary decomposition structure. */
	unsigned char ISads;

	/** String defining the arbitrary decomposition structure. */
	unsigned char* DSads;
};

struct type_atk {
	/** Index of marker segment */
	unsigned char index;

	/** Coefficients data type */
	unsigned char coeff_type;

	/** Wavelet filters */
	unsigned char filter_category;

	/** Wavelet type */
	unsigned char wavelet_type;

	/** Odd/Even indexed subsequence */
	unsigned char m0;

	/** Number of lifting steps */
	unsigned char lifing_steps;

	/** Number of lifting coefficients at lifting step */
	unsigned char lifting_coefficients_per_step;

	/** Offset for lifting step */
	unsigned char lifting_offset;

	/** Base two scaling exponent for lifting step s, εs for the reversible transform only */
	unsigned char* scaling_exponent;
	
	/** Scaling factor, for the irreversible transform only*/
	unsigned char* scaling_factor;

	/**The ith lifting coefficient for the jth lifting step,αs,k. The index, i, ranges from i = 0 to Natk-1 and is the inner loop (present for all of j). The index, j, ranges from j = 0 to Latk-1 and is the outer loop(incremented after a full run of i). */
	unsigned char * coefficients;

	/** The ith additive residue for lifting step, s. The index, i, ranges from i = 0 to Natk-1. Present for reversible transformations */
	unsigned char* additive_residue;
};

#endif
