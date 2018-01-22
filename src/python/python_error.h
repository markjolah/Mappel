/** @file python_error.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 01-2018
 * @brief Definitions for the py11_armadillo namespace, numpy to armadillo conversions
 */

#ifndef _PYTHON_ERROR_H
#define _PYTHON_ERROR_H

#include <exception>
#include <string>

class PythonError : public std::exception
{
public:

    /** @brief Create a PythonError with specified condition
    */
    PythonError(std::string message) :  _condition("C++ Interface Error"), _message(message) {};
    PythonError(std::string condition, std::string message) :  _condition(condition), _message(message) {};
    const char* condition() const noexcept { return _condition.c_str(); };
    const char* message() const noexcept { return _message.c_str(); };
    const char* what() const noexcept override { return (_condition+": "+_message).c_str(); };
protected:
    std::string _condition;
    std::string _message;
};

#endif /*_PYTHON_ERROR_H */
