// Copyright (c) 2009-2013 Intel Corporation
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


#ifndef _INTEL_OPENCL_SAMPLE_BASIC_HPP_
#define _INTEL_OPENCL_SAMPLE_BASIC_HPP_


#include <cstdlib>
#include <cassert>
#include <string>
#include <stdexcept>
#include <sstream>
#include <typeinfo>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <exception>
#include <CL/cl.h>

using std::string;


// Returns textual representation of the OpenCL error code.
string opencl_error_to_str (cl_int error);


// Base class for all exception in samples
class Error : public std::runtime_error
{
public:
    Error (const string& msg) :
        std::runtime_error(msg)
    {
    }
};


// Allocates piece of aligned memory
// alignment should be a power of 2
// Out of memory situation is reported by throwing std::bad_alloc exception
void* aligned_malloc (size_t size, size_t alignment);

// Deallocates memory allocated by aligned_malloc
void aligned_free (void *aligned);


// Represent a given value as a string and enclose in quotes
template <typename T>
string inquotes (const T& x, const char* q = "\"")
{
    std::ostringstream ostr;
    ostr << q << x << q;
    return ostr.str();
}


// Convert from a string to a value of a given type.
// T should have operator>> defined to be read from stream.
template <typename T>
T str_to (const string& s)
{
    std::istringstream ss(s);
    T res;
    ss >> res;

    if(!ss || (ss.get(), ss))
    {
        throw Error(
            "Cannot interpret string " + inquotes(s) +
            " as object of type " + inquotes(typeid(T).name())
        );
    }

    return res;
}


// Convert from a value of a given type to string with optional formatting.
// T should have operator<< defined to be written to stream.
template <typename T>
string to_str (const T x, std::streamsize width = 0, char fill = ' ')
{
    using namespace std;
    ostringstream os;
    os << setw(width) << setfill(fill) << x;
    if(!os)
    {
        throw Error("Cannot represent object as a string");
    }
    return os.str();
}


// Report about an OpenCL problem.
// Macro is used instead of a function here
// to report source file name and line number.
#define SAMPLE_CHECK_ERRORS(ERR)                        \
    if(ERR != CL_SUCCESS)                               \
    {                                                   \
        throw Error(                                    \
            "OpenCL error " +                           \
            opencl_error_to_str(ERR) +                  \
            " happened in file " + to_str(__FILE__) +   \
            " at line " + to_str(__LINE__) + "."        \
        );                                              \
    }


// Detect if x is string representation of int value.
bool is_number (const string& x);


// Return one random number uniformally distributed in
// range [0,1] by std::rand.
// T should be a floatting point type
template <typename T>
T rand_uniform_01 ()
{
    return T(std::rand())/RAND_MAX;
}


// Fill array of a given size with random numbers
// uniformally distributed in range of [0,1] by std::rand.
// T should be a floatting point type
template <typename T>
void fill_rand_uniform_01 (T* buffer, size_t size)
{
    std::generate_n(buffer, size, rand_uniform_01<T>);
}


// Returns random index in range 0..n-1
inline size_t rand_index (size_t n)
{
    return static_cast<size_t>(std::rand()/((double)RAND_MAX + 1)*n);
}


// Returns current system time accurate enough for performance measurements
double time_stamp ();

// Follows safe procedure when exception in destructor is thrown.
void destructorException ();


// Query for several frequently used device/kernel capabilities

// Minimal alignment in bytes for memory used in clCreateBuffer with CL_MEM_USE_HOST_PTR
cl_uint requiredOpenCLAlignment (cl_device_id device);

// Maximum number of work-items in a workgroup
size_t deviceMaxWorkGroupSize (cl_device_id device);

// Maximum number of work-items that can be
// specified in each dimension of the workgroup
void deviceMaxWorkItemSizes (cl_device_id device, size_t* sizes);

// Maximum work-group size that can be used to execute
// a kernel on a specific device
size_t kernelMaxWorkGroupSize (cl_kernel kernel, cl_device_id device);


// Returns directory path of current executable.
string exe_dir ();


double eventExecutionTime (cl_event event);


#endif  // end of include guard
