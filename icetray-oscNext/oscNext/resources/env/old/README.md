# Deploy IceTray (GitHub) to CVMFS

Use singularity container to map paths to CVMFS
```
singularity exec -c --writable-tmpfs -B /tmp:/tmp -B /cvmfs:/cvmfs -B /net/cvmfs_users:/cvmfs/icecube.opensciencegrid.org/users /cvmfs/singularity.opensciencegrid.org/opensciencegrid/osgvo-el7:latest bash
```

Clone the repo:
```
cd /cvmfs/icecube.opensciencegrid.org/users/stuttard/icetray/oscNext
git clone -b oscNext git@github.com:icecube/icetray.git src
```

**HACK:** When I am in the singularity container, cmake cannot find OpenCL. I added this hack to `src/cmake/tools/opencl.cmake` and it seems to work, need a better solution though.
```
 # Hack added by Tom to make it work when deploying to CVMFS user area (just above the line 'find_package(OpenCL QUIET)')
 SET(OPENCL_BASE /cvmfs/icecube.opensciencegrid.org/distrib/OpenCL_RHEL_7_x86_64)
 SET(OpenCL_LIBRARY "${OPENCL_BASE}/lib")
 SET(OpenCL_INCLUDE_DIR "${OPENCL_BASE}/include")
```

Build:
```
eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/setup.sh`
mkdir build_py3-v4.2.1
ln -s build_py3-v4.2.1 build
cd build_py3-v4.2.1
cmake ../src
make -j 12
```

Install external dependenceis post-build:
```
./env-shell.sh # Need to load env first
export PYTHONUSERBASE=$I3_BUILD/local # Point pip --user to the icetray build dir
sh $I3_SRC/oscNext/resources/env/install_metaproject_deps.sh
```

Done! Remember that the sync is hourly, so might not see changes right away.
