/*
# Copyright (c) 2014-2015, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef NVXIO_UTILITY_HPP
#define NVXIO_UTILITY_HPP

#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <string>

#include <NVX/nvx.h>

/**
 * \file
 * \brief The `NVXIO` utility functions.
 */

namespace nvxio
{
/**
 * \defgroup group_nvxio_utility Utility
 * \ingroup nvx_nvxio_api
 *
 * Defines NVXIO Utility API.
 */

// Auxiliary macros

/**
 * \ingroup group_nvxio_utility
 * \brief Throws `std::runtime_error` exception.
 * \param [in] msg A message with content related to the exception.
 * \see nvx_nvxio_api
 */
#define NVXIO_THROW_EXCEPTION(msg) \
    do { \
        std::ostringstream ostr_; \
        ostr_ << msg; \
        throw std::runtime_error(ostr_.str()); \
    } while(0)

/**
 * \ingroup group_nvxio_utility
 * \brief Performs an operation. If the operation has failed then it throws `std::runtime_error` exception.
 * \param [in] vxOp A function to be called.
 * The function must have `vx_status` return value.
 * \see nvx_nvxio_api
 */
#define NVXIO_SAFE_CALL(vxOp) \
    do \
    { \
        vx_status status = (vxOp); \
        if (status != VX_SUCCESS) \
        { \
            NVXIO_THROW_EXCEPTION(# vxOp << " failure [status = " << status << "]" << " in file " << __FILE__ << " line " << __LINE__); \
        } \
    } while (0)

/**
 * \ingroup group_nvxio_utility
 * \brief Checks a condition. If the condition is false then it throws `std::runtime_error` exception.
 * \param [in] cond Expression to be evaluated.
 * \see nvx_nvxio_api
 */
#define NVXIO_ASSERT(cond) \
    do \
    { \
        bool status = (cond); \
        if (!status) \
        { \
            NVXIO_THROW_EXCEPTION(# cond << " failure in file " << __FILE__ << " line " << __LINE__); \
        } \
    } while (0)

/**
 * \ingroup group_nvxio_utility
 * \brief Checks a reference. If the reference is not valid then it throws `std::runtime_error` exception.
 * \param [in] ref Reference to be checked.
 * \see nvx_nvxio_api
 */
#define NVXIO_CHECK_REFERENCE(ref) \
    NVXIO_ASSERT(ref != 0 && vxGetStatus((vx_reference)ref) == VX_SUCCESS)

/**
 * \ingroup group_nvxio_utility
 * \brief Returns the size of an array (the \p N template argument).
 * \see nvx_nvxio_api
 */
template <typename T, vx_size N>
vx_size dimOf(T (&)[N]) { return N; }

static std::string toStr(const vx_status& status)
{
    std::string error;
    switch(status)
    {
        case VX_FAILURE:
            error = "VX GENERAL FAILURE ";
            break;
        case VX_ERROR_INVALID_REFERENCE:
            error = "VX INVALID REFERENCE ";
            break;
        case VX_ERROR_INVALID_PARAMETERS:
            error = "VX INVALID PARAMETER ";
            break;
        default:
            error = "NOT DEFINED ERROR";
             break;
    }

    return error;
}


// Common constants

/**
 * \ingroup group_nvxio_utility
 * \brief Double-precision PI.
 * \see nvx_nvxio_api
 */
const vx_float64 PI = 3.1415926535897932;
/**
 * \ingroup group_nvxio_utility
 * \brief Float-precision PI.
 * \see nvx_nvxio_api
 */
const vx_float32 PI_F = 3.14159265f;

// Auxiliary functions

/**
 * \ingroup group_nvxio_utility
 * \brief The callback for OpenVX error logs, which prints messages to standard output.
 * Must be used as a parameter for \ref vxRegisterLogCallback.
 * \param [in] context  Specifies the OpenVX context.
 * \param [in] ref      Specifies the reference to the object that generated the error message.
 * \param [in] status   Specifies the error code.
 * \param [in] string   Specifies the error message.
 */
void VX_CALLBACK stdoutLogCallback(vx_context context, vx_reference ref, vx_status status, const vx_char string[]);

/**
 * \ingroup group_nvxio_utility
 * \brief Checks whether the context is valid and throws an exception in case of failure.
 * \param [in] context Specifies the context to check.
 * \see nvx_nvxio_api
 */
void checkIfContextIsValid(vx_context context);

/**
 * \ingroup group_nvxio_utility
 * \brief `%ContextGuard` is a wrapper for `vx_context`. It is intended for safe releasing of some resources.
 * It is recommended to use `ContextGuard` in your OWR application instead of `vx_context`.
 * \see nvx_nvxio_api
 */
struct ContextGuard
{
    ContextGuard() : context(vxCreateContext()) {
        checkIfContextIsValid(context);
    }
    ContextGuard(const ContextGuard &) = delete;
    ContextGuard &operator = (const ContextGuard &) = delete;
    ~ContextGuard() {
        vxReleaseContext(&context);
    }
    operator vx_context() { return context; }

private:
    vx_context context;
};

/**
 * \ingroup group_nvxio_utility
 * \brief make_unique function.
 * \see nvx_nvxio_api
 */
template <typename T, typename... Args>
std::unique_ptr<T> makeUP(Args &&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

}

#endif // NVXIO_UTILITY_HPP

