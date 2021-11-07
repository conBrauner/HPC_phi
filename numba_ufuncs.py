import numba as nb
from numpy.matrixlib.defmatrix import matrix

@nb.vectorize([nb.float32(nb.float32, nb.float32)])
def hadamard(element_1, element_2):
    return element_1*element_2

@nb.vectorize([nb.float32(nb.float32, nb.bool_, nb.float32)])
def condit_assign_num(matrix_element, mask_element, val):
    if mask_element:
        return val
    else:
        return matrix_element

@nb.vectorize([nb.float32(nb.float32, nb.float32)])
def scalar_add(matrix_element, scalar):
    return matrix_element + scalar

@nb.vectorize([nb.int64(nb.int64, nb.int64)])
def integer_add(integer_1, integer_2):
    return integer_1 + integer_2

@nb.vectorize([nb.bool_(nb.bool_, nb.bool_, nb.bool_)])
def condit_assign_bool(matrix_element, mask_element, val):
    if mask_element:
        return val
    else:
        return matrix_element

@nb.vectorize([nb.float32(nb.float32, nb.bool_, nb.float32)])
def condit_add(matrix_element, mask_element, val):
    if mask_element:
        return matrix_element + val
    else:
        return matrix_element

@nb.vectorize([nb.float32(nb.float32, nb.float32, nb.int32, nb.float32, nb.float32, nb.float32)])
def ornstein_uhlenbeck(xi, mu, tau_xi, dt, weiner, random_number):
    return (mu - xi)*tau_xi*dt + weiner*random_number 

@nb.vectorize([nb.float32(nb.float32, nb.bool_, nb.float32, nb.float32, nb.float32, nb.float32, nb.float32, nb.float32)])
def subthresh_SASLIF_integrate(V, mask_element, dt, rest_V, forcing, W, tau_m, xi):
    if mask_element:
        return V + dt*((-(V - rest_V) - W + xi + forcing)/tau_m)
    else:
        return V 

@nb.vectorize([nb.float32(nb.float32, nb.float32, nb.bool_)])
def test_moperation(elem1, elem2, mask_element):
    if mask_element:
        return elem1 + elem2
    else:
        return elem1

@nb.vectorize([nb.int64(nb.int64, nb.int64)])
def stack_topography_flatten(element_1, element_2):
    return element_1*element_2
