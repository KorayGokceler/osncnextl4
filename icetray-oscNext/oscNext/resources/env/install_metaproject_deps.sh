#!/usr/bin/env bash

# Install external dependencies AFTER the initital icetray build
# Specifically includes setup for graphnet, that include assumption that a virtualenv is being used
# Tom Stuttard

#
# Parse args
#

INSTALL_GRAPHNET=""

while getopts g flag
do
    case "${flag}" in
        g) INSTALL_GRAPHNET="yes";;
    esac
done
echo "INSTALL_GRAPHNET: $INSTALL_GRAPHNET";


#
# Checks
#

# Check if env is setup
if [[ -z "${I3_BUILD}" ]]
then
  echo "ERROR : I3_BUILD not set, must run env-shell.sh before running this script"

else

    # Check virtualenv is activated
    if [[ -z "${VIRTUAL_ENV}" ]]
    then
        echo "ERROR : virtualenv is not activated"
    else


        #
        # Install external python dependencies
        #

        # Note that since using a virtualenv, will NOT user pip --user or pip --target

        # Define a place for cloning external packages into
        EXT_DIR=$I3_BUILD/../ext
        if [ ! -d $EXT_DIR ] 
        then
            mkdir $EXT_DIR
        fi

        # Define a tmp dir, useful for CVMFS installation where working directory disk space may be tight
        # Make sure it exists, otherwise will default to /tmp
        PIP_TMPDIR=$I3_BUILD/pip_tmp
        mkdir $PIP_TMPDIR

        # Install some python dependencies
        #   six, future - py2/3 compatibility handling (required by many `wg-oscillations-fridge` scripts).
        #   uncertainties - dependency of `wg-oscillations-fridge` plotting tools.
        #   uproot - required for genie-reader (must be v4 or greater, as API changes)
        #   lightgbm - required for BDT-based classifiers (e.g. L4 and L7). Enforcing specific version since ML packages to help reproducibility of models/predictions
        TMPDIR=$PIP_TMPDIR pip install --upgrade future six uncertainties "uproot>=4" lightgbm==4.3.0

        # Install nuflux
        TMPDIR=$PIP_TMPDIR CMAKE_PREFIX_PATH=${SROOT} BOOST_ROOT=${SROOT} pip install git+https://github.com/icecube/nuflux


        #
        # Install graphnet
        #

        # Only if user requested it, since rather heavy
        if [[ -n "${INSTALL_GRAPHNET}" ]]
        then

            echo "Installing GraphNeT..."

            # Clone it locally and install from source...
            # Use oscNext branch from ts4051 fork   - TODO specify version
            GRAPHNET_DIR=$EXT_DIR/graphnet
            if [ -d $GRAPHNET_DIR ] 
            then
                echo "graphnet already cloned, skipping (manually remove it from $GRAPHNET_DIR if want a fresh clone)" 
            else
                git clone -b oscNext git@github.com:ts4051/graphnet.git $GRAPHNET_DIR
            fi

            # Had to manually install some dependencies for graphnet first:
            #  1) setuptools
            #  2) cloudpickle (pip dependency resolver issues when installing setuptools)
            #  3) asdf-standard asdf-transform-schemas jmespath imageio PyWavelets tifffile jsonschema (pip dependency resolver issues when installing graphnet)
            #  4) tables (issues at runtime: "ImportError: cannot import name typeDict")
            #  5) Have to manually specify torch_geometric vesion to avoid "from torch_geometric.utils.homophily import homophily" error 
            TMPDIR=$PIP_TMPDIR pip install --upgrade cloudpickle setuptools asdf-standard asdf-transform-schemas jmespath imageio PyWavelets tifffile jsonschema==3.0.2 tables torch_geometric==2.4.0

            # Now install graphnet itself
            cd $GRAPHNET_DIR
            TMPDIR=$PIP_TMPDIR pip install --upgrade -r requirements/torch_gpu.txt -e .[develop,torch]

        else 
            echo "\n>>> NOT installing GraphNeT<<<\n"
        fi


    fi

    #
    # Done
    #

    # Remove the tmp dir
    [ -d $PIP_TMPDIR ] && rm -rf $PIP_TMPDIR

    echo "Done!"


fi

