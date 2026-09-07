#!/usr/bin/env bash
# run_all_examples.sh

# This script runs all the examples in the required order.

# This suite of examples follows the recommended pattern of real-life
# pybdt usage:
#
# - get your data into pybdt.DataSet objects, and pickle them
# - use a simple wrapper for the included resources/scripts/train.py
#   script to train your BDT(s)
# - set up the pybdt.validate.Validator object (which includes
#   applying the BDT to each data set and storing the scores)
# - run one or more validation script to generate BDT testing plots

echo
echo "* Running sample (fake) data generation script..."
$I3_SRC/pybdt/resources/examples/generate_sample_data.py
if [ $? -ne '0' ]; then
    exit 1
fi
echo "* Completed with exit status" ${?}.

echo
echo "* Running training script..."
$I3_SRC/pybdt/resources/examples/train_sample_bdt.sh
if [ $? -ne '0' ]; then
    exit 1
fi
echo "* Completed with exit status" ${?}.

echo
echo "* Running Validator setup script..."
$I3_SRC/pybdt/resources/examples/setup_sample_validator.py
if [ $? -ne '0' ]; then
    exit 1
fi
echo "* Completed with exit status" ${?}.

echo
echo "* Running Validator plotting script..."
$I3_SRC/pybdt/resources/examples/validate_sample_bdt.py
if [ $? -ne 0 ]; then
    exit 1
fi
echo "* Completed with exit status" ${?}.

echo
echo "* Generating an i3 file with sample (fake) data, including BDT scores..."
$I3_SRC/pybdt/resources/examples/pybdt-i3module-example.py
if [ $? -ne 0 ]; then
    exit 1
fi
echo "* Completed with exit status" ${?}.

echo
echo "Done. Output will be available in these directories:"
echo $I3_BUILD/pybdt/resources/examples/data
echo $I3_BUILD/pybdt/resources/examples/output

exit 0
