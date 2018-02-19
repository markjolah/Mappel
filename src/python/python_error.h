/** @file python_error.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 01-2018
 * @brief Definitions for the py11_armadillo namespace, numpy to armadillo conversions
 */

#ifndef _PYTHON_ERROR_H
#define _PYTHON_ERROR_H

#include <exception>
#include <string>

namespace python_error
{

class PythonError : public std::exception
{
public:

    /** @brief Create a PythonError with specified condition
    */
    PythonError(std::string message) :  _condition("C++ Interface Error"), _message(message) {};
    PythonError(std::string condition, std::string message) :  _condition(condition), _message(message), _what(condition+": "+message) {};
    const char* condition() const noexcept { return _condition.c_str(); };
    const char* message() const noexcept { return _message.c_str(); };
    const char* what() const noexcept override { return _what.c_str(); };
protected:
    std::string _condition;
    std::string _message;
    std::string _what;
};

/**
 * Probably should only be called once per module?
 */ 
inline
void register_exceptions()
{
    pybind11::register_exception_translator([](std::exception_ptr err_ptr) {
        try { if(err_ptr) std::rethrow_exception(err_ptr); }
        catch (const PythonError &err) { PyErr_SetString(PyExc_ValueError, err.what()); }
        });
}

} /* namespace python_error */

#endif /*_PYTHON_ERROR_H */
