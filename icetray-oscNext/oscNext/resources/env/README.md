# Installation

The following instructions list how to install IceTray on the current working machine (e.g. NOT to CVMFS).

Optionally you can include GraphNeT as part of this installation.

The use of virtualenv is important for adding non-IceTray packages.

1) Clone the repo:
```
cd <your/install/root/dir>
git clone -b oscNext git@github.com:icecube/icetray.git icetray/oscNext/src
```

2) Build IceTray:

```
eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/setup.sh`
cd icetray/oscNext
mkdir build_py3-v4.2.1
cd build_py3-v4.2.1
cmake ../src
make -j 4
./env-shell.sh
```

3) Install external dependencies:
```
python -m venv venv # The name "venv" is expected by scripts, do not change it
source venv/bin/activate
sh $I3_SRC/oscNext/resources/env/install_metaproject_deps.sh -g
```

Note the `-g` arg means GraphNeT will be included in the installation. If you don't need this (e.g. MC production), can omit this for simplicity.




# Deploy IceTray (GitHub) to CVMFS

This is how to install IceTray in a way that will be deployed to CVMFS (for use with the grid)

1) Use singularity container to map paths to CVMFS
```
singularity exec -c --writable-tmpfs -B /tmp:/tmp -B /cvmfs:/cvmfs -B /net/cvmfs_users:/cvmfs/icecube.opensciencegrid.org/users /cvmfs/singularity.opensciencegrid.org/opensciencegrid/osgvo-el7:latest bash
```

2) Clone the repo:
```
cd /cvmfs/icecube.opensciencegrid.org/users/<your user>/<your path>
git clone -b oscNext git@github.com:icecube/icetray.git src
```

3) Hack OpenCL

**HACK:** When I am in the singularity container, cmake cannot find OpenCL. I added this hack to `src/cmake/tools/opencl.cmake` and it seems to work, need a better solution though.
```
 # Hack added by Tom to make it work when deploying to CVMFS user area (just above the line 'find_package(OpenCL QUIET)')
 SET(OPENCL_BASE /cvmfs/icecube.opensciencegrid.org/distrib/OpenCL_RHEL_7_x86_64)
 SET(OpenCL_LIBRARY "${OPENCL_BASE}/lib")
 SET(OpenCL_INCLUDE_DIR "${OPENCL_BASE}/include")
```

4) Build IceTray and install external depencies as described in the section above...
