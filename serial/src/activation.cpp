
#include "activation.hpp"


/**
 * Safe implementation of sigmoid. This implementation gives a more precise result.
 * 
 * @param[in] x this is the double precision floating
 *              point variable to be filtered by the sigmoid
 * 
 * @return the filtered value
 * 
 * @note    The std::pow() function is low on performance.
 */
double sigmoid(double x)
{
    if (x > 45.0)
    {
        return 1.0;
    }
    else if (x < -45.0)
    {
        return 0.0;
    }
    else
    {
        return 1.0 / (1.0 + std::pow(EXP, -x));
    }
}

/**
 * Computes the derivative of the sigmoid function.
 * This implementation uses already filtered values
 * by the sigmoid to speed up the computation process.
 *
 * @param[in] x this is the double precision floating
 *              point variable to be differentiated
 *
 * @return the derivative of x with respect to the sigmoid function
 */
double sig_derivative(double x)
{
    return (x * (1.0 - x));                 /// Sigmoid derivative formula
}
