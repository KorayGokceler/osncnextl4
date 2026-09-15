'''
Tools for scaling data

Tom Stuttard
'''

class Scaling(object) :
    '''
    Define a scaling to map a data to array to the range [0,1].
    This is useful for minimizers, machine learning, etc.
    Note that an array extending beyond this range can be scaled using this class.
    '''

    def __init__(self,min_val,max_val) :
        self.min_val = min_val
        self.max_val = max_val
        assert self.min_val < self.max_val
        self.gradient = self.max_val - self.min_val
        self.intercept = self.min_val

    def scale(self,var_array) :
        # assert np.nanmin(var_array) >= self.min_val, "%0.3g >= %0.3g" % (np.min(var_array),self.min_val)
        # assert np.nanmax(var_array) <= self.max_val, "%0.3g <= %0.3g" % (np.max(var_array),self.max_val)
        return ( var_array - self.intercept ) / self.gradient 

    def unscale(self,var_array) :
        # assert np.nanmin(var_array) >= 0.
        # assert np.nanmax(var_array) <= 1.
        return ( var_array * self.gradient ) + self.intercept
