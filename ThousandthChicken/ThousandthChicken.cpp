// Copyright (c) 2009-2011 Intel Corporation
// All rights reserved.
//
// WARRANTY DISCLAIMER
//
// THESE MATERIALS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THESE
// MATERIALS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Intel Corporation is the author of the Materials, and requests that all
// problem reports or change requests be submitted to it directly

#include "stdafx.h"

#include <iostream>
#include "basic.hpp"
#include "cmdparser.hpp"
#include "oclobject.hpp"
#include "utils.h"

// for perf. counters
#include <Windows.h>

using namespace std;

extern int decode(OpenCLBasic& oclObjects);
#define PAD_LINES 2

void generateInput(cl_uint* p_input, size_t width, size_t height)
{

    srand(12345);

    // random initialization of input
    for (cl_uint i = 0; i <  width * (height+4); ++i)
    {
        p_input[i] = (rand() | (rand()<<15) ) & 0xFFFFFF;
    }
}

// inline functions for reference kernel
#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

__forceinline unsigned c4max(const unsigned& l, const unsigned& r)
{
    unsigned ur;
    unsigned char* res = (unsigned char*)&ur;
    unsigned char* left = (unsigned char*)&l;
    unsigned char* right = (unsigned char*)&r;

    res[0] = max(left[0],right[0]);
    res[1] = max(left[1],right[1]);
    res[2] = max(left[2],right[2]);
    res[3] = max(left[3],right[3]);

    return ur;
};

__forceinline unsigned c4min(const unsigned& l, const unsigned& r)
{
    unsigned ur;
    unsigned char* res = (unsigned char*)&ur;
    unsigned char* left = (unsigned char*)&l;
    unsigned char* right = (unsigned char*)&r;

    res[0] = min(left[0],right[0]);
    res[1] = min(left[1],right[1]);
    res[2] = min(left[2],right[2]);
    res[3] = min(left[3],right[3]);

    return ur;
};


float ExecuteMedianFilterKernel(cl_uint* p_input, cl_uint* p_output, cl_int width, cl_uint height, OpenCLBasic& ocl, OpenCLProgramOneKernel& executable)
{
    LARGE_INTEGER perf_freq;
    LARGE_INTEGER perf_start;
    LARGE_INTEGER perf_stop;

    cl_int err = CL_SUCCESS;
    cl_uint numStages = 0;

	///////////////////////////////////////////////////////////////////////////////
	//allocate device memory (attached to host memory) and connect to kernel args
    // allocate the buffer with some padding (to avoid boundaries checking)
    cl_mem cl_input_buffer = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_uint) * width * (height+2*PAD_LINES), p_input, &err);
    SAMPLE_CHECK_ERRORS(err);
    if (cl_input_buffer == (cl_mem)0)
        throw Error("Failed to create Input Buffer!");

    // allocate the buffer with some padding (to avoid boundaries checking)
    cl_mem cl_output_buffer = clCreateBuffer(ocl.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_uint) * width * (height+2*PAD_LINES), p_output, &err);
    SAMPLE_CHECK_ERRORS(err);
    if (cl_output_buffer == (cl_mem)0)
        throw Error("Failed to create Output Buffer!");

    err = clSetKernelArg(executable.kernel, 0, sizeof(cl_mem), (void *) &cl_input_buffer);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 1, sizeof(cl_mem), (void *) &cl_output_buffer);
    SAMPLE_CHECK_ERRORS(err);
	//////////////////////////////////////////////////////////////////////////////////////////////



    size_t global_work_size[2] = { width, height};
    size_t offset[2]= { 0, PAD_LINES};

    // execute kernel
    QueryPerformanceCounter(&perf_start);
    err = clEnqueueNDRangeKernel(ocl.queue, executable.kernel, 2, offset, global_work_size, NULL, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(ocl.queue);
    SAMPLE_CHECK_ERRORS(err);
    QueryPerformanceCounter(&perf_stop);

    void* tmp_ptr = NULL;
    tmp_ptr = clEnqueueMapBuffer(ocl.queue, cl_output_buffer, true, CL_MAP_READ, 0, sizeof(cl_uint) * width * (height+2*PAD_LINES), 0, NULL, NULL, NULL);
    if(tmp_ptr!=p_output)
    {
        throw Error("clEnqueueMapBuffer failed to return original pointer");
    }

    err = clFinish(ocl.queue);
    SAMPLE_CHECK_ERRORS(err);

    err = clEnqueueUnmapMemObject(ocl.queue, cl_output_buffer, tmp_ptr, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);

    err = clReleaseMemObject(cl_input_buffer);
    SAMPLE_CHECK_ERRORS(err);
    err = clReleaseMemObject(cl_output_buffer);
    SAMPLE_CHECK_ERRORS(err);

    // retrieve perf. counter frequency
    QueryPerformanceFrequency(&perf_freq);
    return (float)(perf_stop.QuadPart - perf_start.QuadPart)/(float)perf_freq.QuadPart;
}

// main execution routine - performs median filtering with 3x3 kernel size
int main (int argc, const char** argv)
{
    //return code
    int ret = EXIT_SUCCESS;
    // pointer to the HOST buffers
    cl_uint* p_input = NULL;
    cl_uint* p_output = NULL;

    try
    {
        // Define and parse command-line arguments.
        CmdParserCommon cmdparser(argc, argv);

        CmdOption<int> param_width(cmdparser,0,"width","<integer>","width of processed image",4096);
        CmdOption<int> param_height(cmdparser,0,"height","<integer>","height of processed image",4096);
        CmdOptionErrors param_max_error_count(cmdparser);

        cmdparser.parse();

        // Immediatly exit if user wanted to see the usage information only.
        if(cmdparser.help.isSet())
        {
            return EXIT_SUCCESS;
        }

        // Create the necessary OpenCL objects up to device queue.
        OpenCLBasic oclobjects(
            cmdparser.platform.getValue(),
            cmdparser.device_type.getValue(),
            cmdparser.device.getValue()
        );

		decode(oclobjects);

		/*
        // Build kernel
        OpenCLProgramOneKernel executable(oclobjects,"MedianFilter.cl","","MedianFilterBitonic");
		
        int width = param_width.getValue();
        int height = param_height.getValue();
        // validate user input parameters
        {
            if(width < 4 || height < 4 || width > 8192 || height > 8192 )
            {
                throw Error("Input size in each dimension should be in the range [4, 8192]!");
            }

            if((width & (width-1)) || (height & (height-1)))
            {
                throw Error("Input size should be (2^N)!");
            }
        }

        printf("Input size is %d X %d\n", width, height);

        // allocate buffers with some padding (to avoid boundaries checking)
        cl_uint dev_alignment = requiredOpenCLAlignment(oclobjects.device);
        printf("OpenCL data alignment is %d bytes.\n", dev_alignment);
        p_input = (cl_uint*)aligned_malloc(sizeof(cl_uint) * width * (height+2*PAD_LINES), dev_alignment);
        p_output = (cl_uint*)aligned_malloc(sizeof(cl_uint) * width * (height+2*PAD_LINES), dev_alignment);

        if(!(p_input && p_output))
        {
            throw Error("Could not allocate buffers on the HOST!");
        }

        // random input
        generateInput(p_input, width, height);
       // SaveImageAsBMP( p_input, (int)width, (int)height, "MedianFilterInput.bmp");

        // median filtering
        printf("Executing OpenCL kernel...\n");
        float ocl_time = ExecuteMedianFilterKernel(p_input, p_output, width, height, oclobjects, executable);
       // SaveImageAsBMP( p_output + PAD_LINES*width, width, height, "MedianFilterOutput.bmp");
        printf("NDRange perf. counter time %f ms.\n", ocl_time*1000);
		*/

    }
    catch(const CmdParser::Error& error)
    {
        cerr
            << "[ ERROR ] In command line: " << error.what() << "\n"
            << "Run " << argv[0] << " -h for usage info.\n";
        ret = EXIT_FAILURE;
    }
    catch(const Error& error)
    {
        cerr << "[ ERROR ] Sample application specific error: " << error.what() << "\n";
        ret = EXIT_FAILURE;
    }
    catch(const exception& error)
    {
        cerr << "[ ERROR ] " << error.what() << "\n";
        ret = EXIT_FAILURE;
    }
    catch(...)
    {
        cerr << "[ ERROR ] Unknown/internal error happened.\n";
        ret = EXIT_FAILURE;
    }

    aligned_free( p_input );
    aligned_free( p_output );

    return ret;
}

