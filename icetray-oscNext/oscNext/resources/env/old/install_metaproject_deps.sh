#!/usr/bin/env bash

# Install external dependencies AFTER the initital icetray build
# Tom Stuttard

# Check if env is setup
if [[ -z "${I3_BUILD}" ]]
then
  echo "ERROR : I3_BUILD not set, must run env-shell.sh before running this script"

else

    #
    # Install external python dependencies
    #

    # Will install python dependencies that are not part of either the metaproject of the CVMFS python environments here
    # Either use:
    # --target=$TARGET_DIR 
    # or
    # --user (after having done export PYTHONUSERBASE=$I3_BUILD/local before running)

    # Installing in $I3_BUILD/lib so that they python will see them
    # TARGET_DIR=$I3_BUILD/lib

    # Check PYTHONUSERBASE 9s set
    if [[ -z "${PYTHONUSERBASE}" ]]
    then
        echo "ERROR : PYTHONUSERBASE not set"
    else

        # Create the python local dir if does not exist 
        if [ ! -d $PYTHONUSERBASE ] 
        then
            mkdir $PYTHONUSERBASE
        fi

        #TODO check PYTHONUSERBASE is set

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

        # Get latest pip
        TMPDIR=$PIP_TMPDIR pip install --upgrade --user pip

        # Install some python dependencies
        #   six, future - py2/3 compatibility handling (required by many `wg-oscillations-fridge` scripts).
        #   uncertainties - dependency of `wg-oscillations-fridge` plotting tools.
        #   uproot - required for genie-reader
        TMPDIR=$PIP_TMPDIR pip install --upgrade --user six future uncertainties "uproot>=4"

        # Install nuflux. This from from a git repo, and needs a special SROOT variable. 
        # Also ran into issues like "ImportError: No module named glob", which are solved using the '--no-build-isolation' pip arg.
        TMPDIR=$PIP_TMPDIR PREFIX=${SROOT} pip install --no-build-isolation --upgrade --user git+https://github.com/icecube/nuflux


        #
        # Install RetroReco
        #

        #TODO Not yet compatible with recent python

        # # Get RetroReco, which lives in its own git repository, and install it via pip (pointed to the src dir)
        # # Using `--upgrade` to make sure changes to the src and propagated to the installation
        # # Note that doing this BEFORE the main requirements install below, as retro installs a buggy matplotlib version which is overwritten below
        # #TODO Make a release of the RetroReco oscNext branch and pip install that directly
        # RETRO_DIR=$EXT_DIR/retro
        # if [ -d $RETRO_DIR ] 
        # then
        #     echo "RetroReco already cloned, skipping (manually remove it from $RETRO_DIR if want a fresh clone)" 
        # else
        #     # git clone https://github.com/IceCubeOpenSource/retro.git $RETRO_DIR
        #     git clone -b oscNext https://github.com/ts4051/retro.git $RETRO_DIR
        # fi
        # # There used to be an llvmlite version that required forcing an earlier version, but no longer seems to be an issue in 2024
        # TMPDIR=$I3_BUILD/pip_tmp pip install -v --upgrade --user $RETRO_DIR


        #
        # Install graphnet
        #

        # Clone it locally and install from source...

        # Clone repo (oscNext branch from ts4051 fork)   - TODO specify version
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
        TMPDIR=$PIP_TMPDIR pip install --upgrade --user cloudpickle setuptools asdf-standard asdf-transform-schemas jmespath imageio PyWavelets tifffile jsonschema==3.0.2 tables

        # Now install graphnet itself
        cd $GRAPHNET_DIR
        TMPDIR=$PIP_TMPDIR pip install --upgrade --user -r requirements/torch_gpu.txt -e .[develop,torch]


        #
        # Done
        #

        # Remove the tmp dir
        [ -d $PIP_TMPDIR ] && rm -rf $PIP_TMPDIR

        echo "Done!"

    fi

fi

