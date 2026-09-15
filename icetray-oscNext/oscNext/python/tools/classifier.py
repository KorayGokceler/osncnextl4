'''
A class for handling machine learning classification tasks
Uses scikit-learn, as well as packages that support that API (e.g. XGBoost, catboost, ...)

Tom Stuttard
'''

import os, sys, collections, datetime, copy

# Handle change over time in `collections` module
try :
    from collections.abc import Mapping, Sequence # Required as of py3.10
except Exception as e :
    from collections import Mapping, Sequence # Required in py2, works for py<3.10


import numpy as np

#from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
from sklearn.preprocessing import binarize

# joblib import depends on version
try :
    import joblib
except ImportError as e :
    from sklearn.externals import joblib


def get_model_file_paths(file_stem) :
    '''
    Helper function to get the paths for the various files produced by the model training
    '''

    # Store model as joblib file, https://scikit-learn.org/stable/modules/model_persistence.html
    model_file_path = file_stem + ".joblib"

    # Store training data as HDF5
    # Also store a distinct version after the pre-processing, for debuggig
    data_file_path = file_stem + ".hdf5"
    preprocessed_data_file_path = file_stem + "_preprocessed.hdf5"

    return model_file_path, data_file_path, preprocessed_data_file_path


class TrainTestData(object) :
    '''
    A class for holding data for the classifier

    Features :
      - Can select test/train splits, including re-splitting with the option of maintaining statistical independence
      - Data serialization to disk (HDF5 file)
      - Weighting 
      - General safety checks
      - Provide hash of data that can be stored in a classifier for data provenance

    TODOs:
      - Is there a more generic sklearn or similar module I can use for this?
    '''
    def __init__(self,
        classes,
        input_variables,
        output_variable,
        weights=None,
        aux_variables=None,
        seed=None,
    ) :

        # Check inputs
        assert isinstance(classes, Mapping)
        assert isinstance(input_variables, Mapping)
        assert isinstance(output_variable, np.ndarray)
        if aux_variables is not None :
            assert isinstance(aux_variables, Mapping)
        if weights is not None :
            assert isinstance(weights, np.ndarray)

        # Store as members
        self._classes = classes
        self._input_variables = self._format_variables_dict(input_variables)
        self._output_variable = output_variable
        self._weights = weights #TODO check/format weights
        self._train_prob = None
        self._aux_variables = aux_variables
        self._seed = seed

        # Init internals
        self._train_mask = None
        self._train_mask_cum = np.zeros((self.num_events,),dtype=bool)

        # Check all array have same size (using output variable as point of comparison)
        assert self.num_events > 0
        for k,a in self._input_variables.items() :
            assert a.size == self.num_events
        if self._aux_variables is not None :
            for k,a in self._aux_variables.items() :
                assert a.size == self.num_events
        if self._weights is not None :
            assert self._weights.size == self.num_events

        # True class handling
        self.reset_true_classes()


    @classmethod
    def init_from_keys(cls,
        classes,
        data,
        input_variable_keys,
        output_variable_key,
        weight_key=None,
        aux_variable_keys=None,
        seed=None,
    ) :
        '''
        Alternative constructor. Instead of passing the data arrays themselves, pass a 
        big block of data (e.g. a HDF5 file produced by `i3_to_analysis`) and the keys 
        of interest and populate that way instead.
        '''

        # Check inputs
        assert isinstance(data, Mapping)

        # Default weight (as written by this oscNext prpject)
        #if weight_key is None :
        #    weight_key = "I3MCWeightDict.final_weight"

        # Get the output variable
        assert output_variable_key in data, "Output variable '%s' not found in data" % output_variable_key
        output_variable = data[output_variable_key]

        # If a list of desired classes was specified, define a mask for getting these events only 
        if classes is None :
            mask = np.ones_like(output_variable).astype(bool)
        else :
            mask = np.zeros_like(output_variable).astype(bool)
            for c in list(classes.values()) :
                mask = mask | (output_variable == c)
            assert mask.sum() > 0, "No events found of the requested classe(s) : %s" % list(classes.keys())

        # Get all input variables
        for k in input_variable_keys :
            assert k in data, "Input variable '%s' not found in data" % k
        input_variables = collections.OrderedDict([ (v,data[v][mask]) for v in input_variable_keys ])

        # Apply the mask to the output variable
        output_variable = output_variable[mask]

        # Get the weights
        if weight_key is None :
            weights = None
        else :
            assert weight_key in data, "Weight '%s' not found in data" % weight_key
            weights = data[weight_key][mask]

        # Get all aux variables
        if aux_variable_keys is None :
            aux_variables = None
        else :
            for k in aux_variable_keys :
                assert k in data, "Aux variable '%s' not found in data" % k
            aux_variables = collections.OrderedDict([ (v,data[v][mask]) for v in aux_variable_keys ])

        # Instantiate the class
        train_test_data = TrainTestData(
            classes=classes,
            input_variables=input_variables,
            output_variable=output_variable,
            weights=weights,
            aux_variables=aux_variables,
            seed=seed,
        ) 

        # True class handling
        #TODO Make persistent?
        train_test_data.reset_true_classes()

        return train_test_data


    @property
    def num_events(self) :
        '''
        Get the number of events
        '''
        return self._output_variable.size


    @property
    def classes(self) :
        return self._classes


    @property
    def hash(self) :
        '''
        Return a hash of the data
        Only considering he input/output variables, no other element of the state
        This is designed to let the classifier know if the user has provided the same data is was trained on

        Reference: https://stackoverflow.com/questions/16589791/most-efficient-property-to-hash-for-numpy-array
        '''

        # Check if hash has already been computed
        if not hasattr(self,"_hash") : # Weird check, because using custom load from HDF5 file

            #
            # Compute hash
            #

            from hashlib import md5 
            hasher = md5()

            # Hash the keys and arrays for all in/output variables
            # Converting arrays to strings for the hashing
            for k,v in self._input_variables.items() :
                hasher.update(k.encode('utf-8'))
                hasher.update(v.tostring())

            hasher.update(self._output_variable.tostring())

            if self._weights is not None :
                hasher.update(self._weights.tostring())

            # Ignoring aux variables for hash

            # Get the hash
            self._hash = hasher.hexdigest()

        # Return hash
        return self._hash


    def split(self, train_fraction, replace=True) :

        assert replace == True, "Independent re-sampling not yet implemented"

        # Init the RNG
        if not hasattr(self,"_random_state") : # Weird check, because using custom load from HDF5 file
            self._random_state = np.random.RandomState(self._seed)

        # Check `train_fraction` format
        # Can be a single value to apply to all events, or can be a dict where different fractions are provided per classes
        if np.isscalar(train_fraction) :
            train_fraction = { k:train_fraction for k in list(self.classes.keys()) }
        assert isinstance(train_fraction, Mapping)

        print(set(list(self.classes.keys())))
        print(set((train_fraction.keys())))
        assert set(list(self.classes.keys())) == set((train_fraction.keys()))
        for frac in list(train_fraction.values()) :
            assert np.isscalar(frac)
            assert ( (frac >= 0.) and (frac <= 1.) ), "`test_size` must be in range [0,1]"

        # Determine probability for each event individual to be in the training sample, based on its class
        self._train_prob = np.full((self.num_events,),np.NaN)
        #event_train_prob = np.full((self.num_events,),np.NaN)
        for class_key,class_train_frac in train_fraction.items() :
            true_class_mask = self._output_variable == self.classes[class_key]
            self._train_prob[true_class_mask] = class_train_frac
            #event_train_prob[true_class_mask] = class_train_frac
        assert np.all(np.isfinite(self._train_prob))

        # Get mask defining the split
        self._train_mask = self._random_state.uniform(0.,1.,size=self.num_events) < self._train_prob #event_train_prob

        # Add to the cumulative mask of all events used
        self._train_mask_cum = self._train_mask_cum & self._train_mask


    def _format_variables_dict(self,variables) :
        '''
        Format the variablee dicts for storage.
        Sorting and using OrderedDict them to enforce the order, as HDF5 does not preserve this.
        Get issues with hashing unless this is done. 
        '''
        return collections.OrderedDict([ (k,variables[k]) for k in sorted(list(variables.keys())) ])


    def _return_masked_data(self, mask, weight_scaling=None) :
        '''
        Return the input and output variables (+ weights) with a mask applied
        Weights are scaled to account for the reduced number of events in the masked set
        '''

        # Check inputs
        if weight_scaling is not None :
            assert self._weights is not None #TODO Maybe need to make weights that all are 1 in this case
            assert self._weights.shape == weight_scaling.shape

        # Get the input variables 
        masked_input_variables = collections.OrderedDict()
        for k,v in self._input_variables.items() :
            masked_input_variables[k] = v[mask]

        # Get the output variables
        masked_output_variable = self._output_variable[mask]

        # Get the weights, if there are any
        # Re-scale the weights so they still produce the same rate despite the reduce number of events due to the mask
        # This ensure that train/test sample weights are still valid
        if self._weights is None :
            masked_weights = None
        else :
            masked_weights = self._weights[mask]
            if weight_scaling is not None :
                masked_weights *= weight_scaling[mask]

        # Get the aux variables 
        if self._aux_variables is None :
            masked_aux_variables = None
        else :
            masked_aux_variables = collections.OrderedDict()
            for k,v in self._aux_variables.items() :
                masked_aux_variables[k] = v[mask]

        # Get the truth class variable
        masked_true_class_variable = self.true_class_variable[mask]

        return masked_input_variables, masked_output_variable, masked_weights, masked_aux_variables, masked_true_class_variable


    def apply_mask(self,mask) :
        '''
        Apply a mask to the internal data
        Events not passing the mask will be removed from this data container
        '''

        #TODO return new stucture rather than modifying this one?

        # Check the mask
        assert isinstance(mask, np.ndarray), "`mask` must be a numpy array"
        assert mask.ndim == 1, "`mask` must 1D"
        assert mask.dtype == bool, "`mask` must a boolean array"
        assert mask.size == self.num_events, "`mask` must be the same length as the data arrays"

        # Apply the masks to the internal data arrays
        self._input_variables = collections.OrderedDict([ (k,a[mask]) for k,a in self._input_variables.items() ])
        self._output_variable = self._output_variable[mask]
        if self._weights is not None :
            self._weights = self._weights[mask]
        if self._aux_variables is not None :
            self._aux_variables = collections.OrderedDict([ (k,a[mask]) for k,a in self._aux_variables.items() ])

        # Also apply to the masks designating the training events
        if self._train_mask is not None :
            self._train_mask = self._train_mask[mask]
        self._train_mask_cum = self._train_mask_cum[mask]
        if self._train_prob is not None :
            self._train_prob = self._train_prob[mask]


        # Check have events left after making the cut
        assert self.num_events > 0, "No events remaining after applying mask"


    @property
    def train_mask(self) :
        return self._train_mask

    @property
    def test_mask(self) :
        return ~self._train_mask


    def get_train_data(self) :
        '''
        Get the input and output variables for the train data.
        The weights will be corrected for the splitting.
        '''
        assert self._train_mask is not None, "Splitting has not yet been performed"
        weight_scaling = 1. / self._train_prob
        return self._return_masked_data(self.train_mask, weight_scaling)


    def get_test_data(self) :
        '''
        Get the input and output variables for the test data
        The weights will be corrected for the splitting.
        '''
        assert self._train_mask is not None, "Splitting has not yet been performed"
        weight_scaling = 1. / (1. - self._train_prob)
        return self._return_masked_data(self.test_mask, weight_scaling)


    def save(self,file_path) :

        import tables
        from icecube.oscNext.tools.i3_to_analysis import create_or_append_to_array

        assert self._train_mask is not None, "Splitting has not yet been performed"

        output_file = tables.open_file(
            filename=file_path,
            mode="w",
            title="Classifier train/test events File"
        )

        input_group = output_file.create_group(output_file.root, "input", "input")
        for k,v in self._input_variables.items() :
            create_or_append_to_array(output_file,input_group,k,v)

        output_group = output_file.create_group(output_file.root, "output", "output")
        create_or_append_to_array(output_file,output_group,"output",self._output_variable)

        if self._weights is not None :
            weights_group = output_file.create_group(output_file.root, "weights", "weights")
            create_or_append_to_array(output_file,weights_group,"weights",self._weights)

        if self._aux_variables is not None :
            aux_group = output_file.create_group(output_file.root, "aux", "aux")
            for k,v in self._aux_variables.items() :
                create_or_append_to_array(output_file,aux_group,k,v)

        misc_group = output_file.create_group(output_file.root, "misc", "misc")
        create_or_append_to_array(output_file,misc_group,"classes_keys",np.asarray(list(self._classes.keys())))
        create_or_append_to_array(output_file,misc_group,"classes_values",np.asarray(list(self._classes.values())))
        create_or_append_to_array(output_file,misc_group,"train_mask",self._train_mask)
        create_or_append_to_array(output_file,misc_group,"train_mask_cum",self._train_mask_cum)
        if self._train_prob is not None :
            create_or_append_to_array(output_file,misc_group,"train_prob",self._train_prob)
        #TODO seed

        output_file.close()


    # Accessors to the variables
    # Not recommended to mess with these unless you know what you're doing
    @property
    def input_variables(self) : 
        return self._input_variables

    @property
    def output_variable(self) : 
        return self._output_variable

    @property
    def aux_variables(self) : 
        return self._aux_variables

    @property
    def weights(self) : 
        return self._weights


    def get_flat_weights(self, var_keys, var_bins) :
        '''
        Create weights that give a flat distribution w.r.t. the specified parameters
        Do this by histogramming in the specified parameter space to get the PDF, and normalise by that.
        '''

        #TODO support interpolation/splining of the histogram to get continuous distributions


        #
        # Get inputs
        #

        assert len(var_keys) == len(var_bins)

        # Find all variable arrays
        var_arrays = []
        for var_key in var_keys :

            # Find the variable
            if var_key in self.aux_variables :
                var_array = self.aux_variables[var_key]
            elif var_key in self.input_variables :
                var_array = self.input_variables[var_key]
            else :
                raise Exception("Variable '%s' not found" % var_key)
            var_arrays.append(var_array)



        #
        # Compute "unweighted" weight
        #

        # Start form NaN weights
        new_weights = np.full_like(self.weights, np.NaN)

        # Get the histogram
        hist, bin_edges = np.histogramdd(var_arrays, bins=var_bins)

        # Loop to get in bin index for every event
        bin_indices = []
        for var_array, var_bin_edges in zip(var_arrays, bin_edges) :

            # Get the bin index
            # Correct for the under/overflow so can use as bin index (np.digitize 
            # bin 0 is underflow, -1 is overflow, e.g. have Nbins+2 possible indices)
            var_indices = np.digitize(var_array, var_bin_edges)
            var_indices[var_indices==len(var_bin_edges)] = 0
            var_indices -= 1
            bin_indices.append(var_indices)

        # Loop over events and set their weight as 1 / bin count (for the bin that event falls in)
        # Leave events outside of binned range as NaN
        for evt_index in np.ndindex(new_weights.shape) :
            bin_index = tuple([ b[evt_index] for b in bin_indices ])
            if np.all(np.array(bin_index) >= 0) :
                new_weights[evt_index] = 1. / float(hist[bin_index])

        return new_weights



    @classmethod
    def load(cls,file_path,keys=None) :
        '''
        Load the train-test data from an hdf5 file

        keys can be specified as a list of items 
        to Load (instead of loading the whole thing)
        '''

        #TODO Document

        import tables

        file_path = str(os.path.expandvars(file_path))

        input_file = tables.open_file(
            filename=file_path,
            mode="r",
        )

        def load_arrays(group,keys=None) :
            arrays = collections.OrderedDict()
            for obj in group._f_walknodes("EArray") :

                if keys is not None:
                    if (obj._v_name in keys):
                        arrays[obj._v_name] = obj.read()
                else:
                    arrays[obj._v_name] = obj.read()

            return arrays

        input_variables = load_arrays(input_file.root.input,keys=keys)
        output_variable = input_file.root.output.output.read()

        if hasattr(input_file.root,"weights") :
            weights = input_file.root.weights.weights.read()
        else :
            weights = None

        if hasattr(input_file.root,"aux") :
            aux_variables = load_arrays(input_file.root.aux,keys=keys)
        else :
            aux_variables = None

        misc = load_arrays(input_file.root.misc)

        classes = collections.OrderedDict()
        for k, v in zip(misc["classes_keys"], misc["classes_values"]) :
            if not isinstance(k, str) :
                k = k.decode('UTF-8') # byte->str conversion for the class key (happens during HDF5 file writing in some versions)
            classes[k] = v

        train_mask = misc["train_mask"]
        train_mask_cum = misc["train_mask_cum"]

        if "train_prob" in misc :
            train_prob = misc["train_prob"]
        else :
            train_prob = None

        # Create the class
        train_test_data = TrainTestData(
            classes=classes,
            input_variables=input_variables,
            output_variable=output_variable,
            weights=weights,
            aux_variables=aux_variables,
            seed=None, #TODO Also store seed
        ) 
        train_test_data._train_mask = train_mask
        train_test_data._train_mask_cum = train_mask_cum
        if train_prob is not None :
            train_test_data._train_prob = train_prob

        input_file.close()

        return train_test_data


    def report(self) :

        # Report of variables
        #TODO
        
        # Report training sample
        print("Training sample :")
        print("  Total : %i events (of %i, %0.3g%%)" % ( self.train_mask.sum(), self.num_events, 100.*float(self.train_mask.sum())/float(self.num_events) ) )
        for class_key,class_val in self.classes.items() :
            class_true_mask = self._output_variable == class_val
            class_train_mask = class_true_mask & self.train_mask
            class_num_events = class_true_mask.sum()
            class_train_percent = 100. * float(class_train_mask.sum()) / float(class_num_events)
            print("  %s : %i events (of %i, %0.3g%%)" % (class_key,class_train_mask.sum(),class_num_events,class_train_percent) )



    def define_true_classes(self, classes, aux_variable_key) :
        '''

        '''
        self._true_classes = classes
        self._true_class_aux_variable_key = aux_variable_key


    def reset_true_classes(self) :
        '''

        '''
        self._true_classes = None
        self._true_class_aux_variable_key = None

    @property
    def true_classes(self) :
        return self._classes if self._true_classes is None else self._true_classes

    @property
    def true_class_variable(self) :
        return self._output_variable if self._true_classes is None else self._aux_variables[self._true_class_aux_variable_key]


class Classifier(object) :
    '''
    Class used for training a classifier and many related tasks.
    Is general, e.g. should be suitable to tasks like "identify background", "PID", etc.
    Uses the scikit-learn framework, plus additional algorithms like XGBoost and catboost

    This lives entirely outside of the IceCube software framework, but `I3Classifier` (below) 
    can be used to load a model created by this classifier and make predictions within IceCube frames.
    '''

    def __init__(self, algorithm=None) :

        #TODO remove `classes`, should get this from the TrainTestData instance

        # Default classifier is XGBoost
        self.algorithm = algorithm
        if self.algorithm is None :
            self.algorithm = "xgboost"

        # Init members that are filled later
        # Careful control of these is needed as we are customising the pickling later using __get/setstate__
        self._model = None
        self._seed = None
        self._training_time = None
        self._input_variable_names = None
        self._output_variable_name = None
        self._train_test_data = None
        self._train_test_data_hash = None
        self._preprocessed_train_test_data = None


    def __getstate__(self) :
        '''
        Return the class state for pickle/joblib to serialize
        Omitting `train_test_data` as it is large (storing it separately as a HDF5 file)
        '''
        return { k:v for k,v in self.__dict__.items() if k not in [ "_train_test_data", "_preprocessed_train_test_data" ] }
        

    def __setstate__(self,state) :
        '''
        Set the class state when instantiating from a pickle/joblib dump
        add a placeholder for the `train_test_data` member that is filled separately from HDF5
        '''
        self.__dict__ = { k:v for k,v in state.items() }
        self.__dict__["train_test_data"] = None 


    #TODO Make a copy function


    def train_model(self,
        train_test_data,
        seed=None,
        class_ratio=None,
        scale_weights=False,
        event_subset_mask=None,
        weights=None,
        **kw
    ) :
        '''
        TODO

        Args :
            event_subset_mask : (optional) Provide mask to select a subset of the total events to be used for training (not the same as picking a test/train sample, instead use to exclude certain types of events from the training)
            alternative_weights : (optional) Provide a set of weights for the events to be used for training (e.g. a different set of weights compared to the ones actully stored in the TrainTestData)
            kwargs : Passed to the model constructor
        '''

        #TODO Add option to store the weight for each event too (`sample_weight` arg)

        #
        # Check inputs
        #

        assert self._model is None, "Currently only supporting training model in a single pass"

        # Store some stuff as members for later use
        self._train_test_data = train_test_data
        self._seed = seed

        # Store the data has
        # This can then be used to check the same test/train data is used in 
        # the future (such as for making plots using the test data)
        self._train_test_data_hash = train_test_data.hash

        # Take a copy of the classes
        # Cannot directly re-use the version in `train_test_data` as when re-load model without the data then still need the classes
        self._classes = copy.deepcopy(train_test_data.classes)


        #
        # Get the training data
        #

        # Perform any required pre-processing on the data
        self._preprocessed_train_test_data = self._preprocess_data(
            class_ratio=class_ratio,
            scale_weights=scale_weights,
            event_subset_mask=event_subset_mask,
            weights=weights,
        )

        # Get the training data
        train_input_variables, train_output_variable, train_weights, _, _ = self.preprocessed_train_test_data.get_train_data()


        #
        # Format training data for sklearn classifier
        #

        # Check output variable contains only the expected classes
        self._check_output_variable(train_output_variable)

        print("Formatting data for classifier (%i input variables)..." % len(train_input_variables))

        # Store the variable names
        self._input_variable_names = list(train_input_variables.keys())
        # self._output_variable_names = train_output_variable.keys()

        # Format in the correct manner for sklearn
        x_train = self._format_input_data(train_input_variables)
#        y_train = self._format_output_data(train_output_variable)
        y_train = train_output_variable
        weights = train_weights


        #
        # Create and train the model
        #

        #TODO May need to start handlign each classifier type differently
        #TODO For example, using each libraries own data container (catboost.Pool, lgb.Dataset, etc), or avoid the sklern wraper altogether
        #TODO Note that some datasets have their own weighting options, e.g. don't then use sample_weight

        print("Training model...")

        #TODO Use GridSearchCV to optimise hyperparams before fitting...
        #TODO See https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65

        # Time the training
        fit_start_time = datetime.datetime.now()

        # Create the the model, using the classifier of choice
        if self.algorithm == "xgboost" : 

            #
            # XGBoost
            #

            from xgboost import XGBClassifier

            self._model = XGBClassifier(  
                model=self._model,
                seed=self._seed,
                **kw
            )

            # Time the training
            fit_start_time = datetime.datetime.now()

            # Fit the model to the training data
            self._model.fit(
                X=x_train,
                y=y_train,
                sample_weight=weights,
            )


        elif self.algorithm == "catboost" : 

            #
            # CatBoost
            #

            from catboost import CatBoostClassifier, Pool

            self._model = CatBoostClassifier(
#                iterations=2, 
#                depth=2, 
#                learning_rate=1, 
#                loss_function='Logloss', 
                **kw)

            # Create the data pool
            pool = Pool(
                data=x_train,
                label=y_train,
                weight=weights,
            )

            # Fit the model to the training data
            #TODO Could use `sample_weights` for weighting here and avoid Pool (for more common treatment w.r.t. other classifiers)
            self._model.fit(
                pool,
            )


        elif self.algorithm == "lightgbm" : 

            #
            # Light GBM
            #

            from lightgbm import LGBMClassifier

            self._model = LGBMClassifier(**kw) # TODO args

            #TODO Use lightgbm.Dataset?

            self._model.fit(
                x_train,
                y_train,
                sample_weight=weights,
            )


        else :
            raise Exception("Unknown classifier algorithm : " % self.algorithm)


        # Report the time
        fit_end_time = datetime.datetime.now()
        fit_time_taken = fit_end_time - fit_start_time
        print("...Training took %s" % fit_time_taken)

        # Increment the total training time counter
        if self._training_time is None :
            self._training_time = fit_time_taken
        else :
            self._training_time += fit_time_taken


    def _preprocess_data(self,
        class_ratio=None,
        scale_weights=False,
        event_subset_mask=None,
        weights=None,
    ) :
        '''
        Function for preparing/pre-processing the training data 
        '''

        #
        # Prepare data
        #

        # Make a copy of the data container
        preprocessed_train_test_data = copy.deepcopy(self.train_test_data)

        # Replace weights with user specified versions
        if weights is not None :
            assert weights.size == preprocessed_train_test_data.num_events
            preprocessed_train_test_data._weights = weights
        
        # Apply the event subset mask
        if event_subset_mask is not None :
            preprocessed_train_test_data.apply_mask(event_subset_mask)



        #
        # Handle normalisation/resampling
        #

        # Re-weight the data (without changing shape) to achieve the desired ratio of events in the sample

        if class_ratio is not None :

            # Check inputs
            # TODO

            # Create null weights if none provided
            if preprocessed_train_test_data.weights is None :
                preprocessed_train_test_data._weights = np.ones(preprocessed_train_test_data.num_events).astype(np.float64)

            # Get weight sum for each class
            class_masks = collections.OrderedDict()
            weight_sums = collections.OrderedDict()
            for class_key, ratio in class_ratio.items() : #TODO enforce self.classes order?
                class_masks[class_key] = preprocessed_train_test_data.output_variable == self.classes[class_key]
                weight_sums[class_key] = np.nansum( preprocessed_train_test_data._weights[class_masks[class_key]] )

            # Adjust weights to observe desired ratio
            ref_class_key = list(self.classes.keys())[0]
            for this_class_key in list(self.classes.keys())[1:] :
                weight_scaling = ( class_ratio[this_class_key] / class_ratio[ref_class_key] ) * ( weight_sums[ref_class_key] / weight_sums[this_class_key] )
                preprocessed_train_test_data._weights[class_masks[class_key]] *= weight_scaling


        #
        # Weight scaling
        #

        # Scale weights to be in the range [0,1]
        # Some algorithms can't cope with huge ranges in weights, or very small weights, or whatever

        if scale_weights :

            assert preprocessed_train_test_data._weights is not None, "Cannot use `scale_weights` if no weights are provided"

            from icecube.oscNext.tools.scaling import Scaling

            weight_scaling = Scaling(min_val=preprocessed_train_test_data._weights.min(), max_val=preprocessed_train_test_data._weights.max())
            preprocessed_train_test_data._weights = weight_scaling.scale(preprocessed_train_test_data._weights)


        #
        # "Unweight" input distributions
        #

        #TODO


        #
        # Done
        #

        return preprocessed_train_test_data



    def report(self) :
        #TODO Put a bit more here, include info on training sample, loop over hyperparamters, training time, etc
        #TODO Option to plot as table too
        print(model)


    def training_data_stats(self) :
        '''
        Get some stats on the training data
        '''

        # Containers to return
        raw_event_counts = collections.OrderedDict()
        rates = collections.OrderedDict()

        # Get the training data
        train_input_variables, train_output_variable, train_weights, _, train_true_class_variable = self.train_test_data.get_train_data()


        #
        # Get event stats
        #

        # Loop over classes
        for class_key in list(self.true_classes.keys()) : 

            # Mask to find events of this class
            class_mask = train_true_class_variable == self.class_value(class_key) 

            # Get the number of events per class
            # Not weighted, raw number
            raw_event_counts[class_key] = class_mask.sum()

            # Get the rate
            if train_weights is not None :
                rates[class_key] = train_weights[class_mask].sum()

        return raw_event_counts,rates


    def testing_data_stats(self) :
        '''
        Get some stats on the testing data
        '''

        # Containers to return
        raw_event_counts = collections.OrderedDict()
        rates = collections.OrderedDict()

        # Get the testing data
        test_input_variables, test_output_variable, test_weights, _, test_true_class_variable = self.train_test_data.get_test_data()


        #
        # Get event stats
        #

        # Loop over classes
        for class_key in list(self.true_classes.keys()) : 

            # Mask to find events of this class
            class_mask = test_true_class_variable == self.class_value(class_key) 

            # Get the number of events per class
            # Not weighted, raw number
            raw_event_counts[class_key] = class_mask.sum()

            # Get the rate
            if test_weights is not None :
                rates[class_key] = test_weights[class_mask].sum()

        return raw_event_counts,rates



    def _check_output_variable(self,output_variable) :
        '''
        Check the output variable
        It must not contain unknown classes
        '''

        class_values_allowed = set(list(self.classes.values()))
        class_values_found = set(np.unique(output_variable))
        assert class_values_found.issubset(class_values_allowed), "Unknown output variable value(s) found : %s" % class_values_found.difference(class_values_allowed)


    def _format_input_data(self,input_variables) :
        '''
        Format input data for sklearn classifier
        Input is a dict of M arrays
        Classifier wants input data in (N,M) array
        N is number of events and M is number of input variables
        '''

        #TODO Is inefficient to repeastedly run this, but is nice for the user to have the data as dicts of arrays. Think about improving this...

        #TODO Use same checking functions as TrainTestData (make them static)

        # Check format of input
        assert isinstance(input_variables, Mapping)

        # Check all variables are present and in the correct order
        # Only do this if the model has already been trained, otherwise don't yet know the variables
        #TODO Could add code to re-order them
        if self.input_variable_names is not None :
            assert self.input_variable_names == list(input_variables.keys()), "Input variables do not match the model variables" #TODO Clearer message

        # Check all variables have the same length
        #TODO

        # Convert to a 2D array as sklearn (or similar interfaces) like it
        return np.stack([ x for x in list(input_variables.values()) ]).T



    @property
    def model(self) :
        return self._model

    @property
    def classes(self) :
        return self._classes

    @property
    def true_classes(self) :
        return self.train_test_data.true_classes

    @property
    def seed(self) :
        return self._seed

    @property
    def training_time(self) :
        return self._training_time

    @property
    def input_variable_names(self) :
        return self._input_variable_names

    @property
    def train_test_data(self) :
        return self._train_test_data

    @property
    def preprocessed_train_test_data(self) :
        return self._preprocessed_train_test_data


    def save(self, file_stem) :
        '''
        Store both the model and also the train/test data.
        Do this as two different files for memory efficiency, and because 
        a user may not want the testing/training data.
        '''

        # Get file paths
        model_file_path, data_file_path, preprocessed_data_file_path = get_model_file_paths(file_stem)

        assert model_file_path.endswith(".joblib")
        assert data_file_path.endswith(".hdf5")
        assert preprocessed_data_file_path.endswith(".hdf5")


        #
        # Store model as joblib file
        #

        print("Saving model to file : %s" % model_file_path)
        joblib.dump(self,model_file_path) 


        #
        # Store the train/test data as HDF5
        #

        print("Saving train/test data to file : %s" % data_file_path)
        self.train_test_data.save(data_file_path)

        if not self.preprocessed_train_test_data is None:
            print("Saving preprocessed train/test data to file : %s" % preprocessed_data_file_path)
            self.preprocessed_train_test_data.save(preprocessed_data_file_path)


    @classmethod
    def load(cls, model_file_path, train_test_data_file_path=None, preprocessed_train_test_data_file_path=None,keys=None) :

        #
        # Load the model/classifier
        #

        model_file_path = os.path.expandvars(model_file_path)
        classifier = joblib.load(model_file_path)
        print("Loaded model from file : %s" % model_file_path)
        assert isinstance(classifier, Classifier), "Loaded model from '%s' is not of type `Classifier`" % model_file_path


        #
        # Load train/test data
        #

        # Check if user provided this
        if train_test_data_file_path is not None :

            # Load the train/test data and add to classifier
            train_test_data_file_path = os.path.expandvars(train_test_data_file_path)
            classifier._train_test_data = TrainTestData.load(train_test_data_file_path,keys=keys)
            print("Loaded train/test data from file : %s" % train_test_data_file_path)
            assert isinstance(classifier.train_test_data,TrainTestData), "Loaded train/test data from '%s' is not of type `TrainTestData`" % train_test_data_file_path

            # Check train./test data is the same as was used to train
            assert classifier._train_test_data_hash == classifier.train_test_data.hash, "Test/train data loaded from file does not match the data used to train the model"

        # Also do the preprocessed version of the test/train data
        if preprocessed_train_test_data_file_path is not None :
            preprocessed_train_test_data_file_path = os.path.expandvars(preprocessed_train_test_data_file_path)
            classifier._preprocessed_train_test_data = TrainTestData.load(preprocessed_train_test_data_file_path)
            print("Loaded preprosessed train/test data from file : %s" % preprocessed_train_test_data_file_path)
            assert isinstance(classifier.train_test_data,TrainTestData), "Loaded preprocessed train/test data from '%s' is not of type `TrainTestData`" % train_test_data_file_path

        return classifier


    def _class_index(self,class_key) :
        '''
        Get the index (e.g. element of the model outputs) for data for the specified class
        Users should not use this directly
        '''
        assert class_key in self.classes, "Unrecognised class %s, choose from %s" % (class_key,self.classes)
        assert self.classes[class_key] in self._model.classes_.tolist(), "Cannot find class '%s' in underlying model (found %s)" % (self.classes[class_key],self._model.classes_.tolist())
        class_idx = self._model.classes_.tolist().index( self.classes[class_key] )
        return class_idx


    def class_value(self,class_key) :
        '''
        Get the numeric value corresponding to the class key
        '''
        for k in self.classes.keys() :
            assert class_key in self.classes, "Unrecognised class %s, choose from %s" % (class_key, list(self.classes.keys()))
        return self.classes[str(class_key)]


    def predict_proba(self, input_variables) :
        '''
        Return a dict containing the predicted probability for every event to be each classifier type
        Basically wraps sklearn `predict_proba` but returns a dict instead for nicer access
        '''
        assert isinstance(input_variables, Mapping)
        x = self._format_input_data(input_variables)
        probs = self._model.predict_proba(x)
        return collections.OrderedDict([ (class_key,probs[:,self._class_index(class_key)]) for class_key in list(self.classes.keys()) ])
                

    def predict(self, input_variables, class_key) :
        '''
        Predict the probability that each event is of class `class_key`
        '''

        # Predict
        y_pred_prob = self.predict_proba(input_variables)

        # Get the probability for this class
        class_prob = y_pred_prob[class_key]

        return class_prob


    def classify(self, input_variables, class_key, threshold) :

        #TODO support non-binary classification

        # Predict probability event is the specified class
        class_prob = self.predict( input_variables=input_variables, class_key=class_key )

        # Classify based on whether probability is above the threshold
        class_id_mask = class_prob >= threshold

        return class_id_mask


    def get_roc_curve(
        self,
        target_class_key,
        input_variables,
        output_variable,
        weights=None,
        thresholds=None,
        target_class_prob=None,
    ) :
        '''
        Create a ROC curve 
        '''

        #TODO Only supports binary classification, make more genera;

        # Checks
        self._check_output_variable(output_variable)

        # Define thresholds if not provide
        if thresholds is None :
            thresholds = np.linspace(0.,1.,num=1000)

        # Use weights, or counts if none provided
        if weights is None :
            weights = np.ones_like(output_variable).astype(float)


        #
        # Get true/false positive rates vs threshold
        #

        trp_values = []
        fpr_values = []

        # Get the model predictions that the events are the target class, if none provided
        if target_class_prob is None :
            target_class_prob = self.predict(input_variables=input_variables, class_key=target_class_key)

        # Define mask indicating if the event actually is the target class
        true_mask = output_variable == self.class_value(target_class_key)

        # Loop over thresholds
        for thr in thresholds :

            # Classify with this threshold
            positive_mask = target_class_prob >= thr

            # True/false positive classification masks
            true_positive_mask = positive_mask & true_mask
            false_positive_mask = positive_mask & (~true_mask)

            # Get the true/false positive rates
            # These are really fractions, but this is what machine learning people mean when they say T/FPR
            tpf = float(weights[true_positive_mask].sum()) / float(weights[true_mask].sum())
            fpf = float(weights[false_positive_mask].sum()) / float(weights[~true_mask].sum())

            # Store
            trp_values.append(tpf)
            fpr_values.append(fpf)

        # Numpy-ify
        trp_values = np.asarray(trp_values)
        fpr_values = np.asarray(fpr_values)

        return thresholds, trp_values, fpr_values


    def get_rates(self, input_variables, output_variable, class_mask, weights=None, ) :
        '''
        Get the rates of each class, given some classification mask (e.g. as returned by Classifier.classify)
        '''

        class_rates = collections.OrderedDict()

        # If not weights provided, just count
        if weights is None :
            weights = np.ones( list(input_variables.values())[0].shape, dtype=np.float64 )

        # Loop over class
        for class_key in list(self.classes.keys()) :

            # Get truth mask, e.g. mask that is True for the requested class
            true_class_mask = output_variable == self.class_value(class_key)

            # Get and store the rate
            rate = weights[true_class_mask&class_mask].sum()
            class_rates[class_key] = rate

        return class_rates



class I3Classifier(object) :
    '''
    Class with I3Module-like interface.
    Use to load an existing `Classifier` model and use within an IceTray 
    to it to predict the class for each frame.
    '''

    def __init__(self,model_file,class_key,output_key) :

        #TODO Overwrite option

        # Store args
        self.class_key = class_key
        self.output_key = output_key

        # Load the model from file
        self.model = Classifier.load(model_file)

        # Keep track that all variables were found
        self.input_variable_counter = collections.OrderedDict([ (k,np.int64(0)) for k in self.model.input_variable_names ])


    def __call__(self,frame) :

        import collections
        from icecube import dataclasses
        from icecube.oscNext.tools.i3_to_analysis import get_frame_variable

        # Check haven't already written to frame
        assert self.output_key not in frame, "Classifier probability already in frame, something has gone wrong"


        #
        # Collect input variables from the frame
        #

        input_variables = collections.OrderedDict()

        # Loop over the model input variables
        for input_variable_name in self.model.input_variable_names :

            # Grab the value from the frame
            value = get_frame_variable(frame,input_variable_name)

            # If can't find it, use NaN #TODO is this correct, or should I just not include it?
            if value is None :
                value = np.NaN

            # Count what we did find
            else :
                self.input_variable_counter[input_variable_name] += 1

            #TODO Add check in case a particular variable is NEVER found

            # Store it as an input for the classsifier
            input_variables[input_variable_name] = np.array([value])

        #TODO Do I need to check if at least some variables are non-NaN???


        #
        # Classify using the model
        #

        # Get the model prediction for whether the event is of the specified class
        prediction_prob = self.model.predict(input_variables,self.class_key)

        # Write the prediction to the frame
        prediction_prob = prediction_prob.astype(np.float64)[0]
        frame[self.output_key] = dataclasses.I3Double(prediction_prob)

        return True

