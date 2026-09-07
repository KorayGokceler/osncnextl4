
i3_project (pybdt
    DOCS_DIR resources/docs
)

if(NUMPY_FOUND)
if(BUILD_PYBDT)
    
i3_add_library (pybdt
    private/pybdt/bdtlearner.cpp
    private/pybdt/bdtmodel.cpp
    private/pybdt/dataset.cpp
    private/pybdt/dtlearner.cpp
    private/pybdt/dtmodel.cpp
    private/pybdt/histogram.cpp
    private/pybdt/learner.cpp
    private/pybdt/linear_histogram.cpp
    private/pybdt/model.cpp
    private/pybdt/nonlinear_histogram.cpp
    private/pybdt/pruner.cpp
    private/pybdt/pybdt.cpp
    private/pybdt/random_sampler.cpp
    private/pybdt/vinelearner.cpp
    private/pybdt/vinemodel.cpp
    USE_TOOLS boost gsl python numpy
    USE_PROJECTS icetray dataclasses
    COMPILE_FLAGS "-DBOOST_ALLOW_DEPRECATED_HEADERS -DBOOST_BIND_GLOBAL_PLACEHOLDERS"
    WITHOUT_I3_HEADERS
    MODULE
    NO_DOXYGEN
    INSTALL_DESTINATION ${Python_PLATLIB}/pybdt
)

i3_test_scripts(
    resources/test/test_ml.py
)
#set_tests_properties(pybdt::run_all_examples.sh
#    pybdt::test_pybdt_api.py
#    PROPERTIES LABELS RUN_LONG
#)

set_target_properties (pybdt
    PROPERTIES 
    PREFIX "_"
    LIBRARY_OUTPUT_DIRECTORY ${PYTHON_PLATLIB_DIR}/pybdt
    SUFFIX ${Python_EXT_SUFFIX}
)

file (MAKE_DIRECTORY "${PYTHON_PLATLIB_DIR}/pybdt")

# install at top level
foreach(pyfile
    py.typed
    __init__.py
    histlite.py
    ml.py
    util.py
    validate.py
    viz.py
)
    execute_process(COMMAND ln -sf ${CMAKE_CURRENT_SOURCE_DIR}/python/${pyfile} ${PYTHON_PLATLIB_DIR}/pybdt/${pyfile})
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/python/${pyfile} DESTINATION ${Python_PLATLIB}/pybdt OPTIONAL)
endforeach()

# install in icecube namespace
foreach(pyfile
    pybdtmodule.py
    convertdsmodule.py
)
    execute_process(COMMAND ln -sf ${CMAKE_CURRENT_SOURCE_DIR}/python/${pyfile} ${PYTHON_PLATLIB_DIR}/icecube/${pyfile})
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/python/${pyfile} DESTINATION ${Python_PLATLIB}/icecube OPTIONAL)
endforeach()

endif(BUILD_PYBDT)

else(NUMPY_FOUND)
    colormsg(CYAN "+-- numpy not found, skipping pybdt")
endif(NUMPY_FOUND)

# vim: nospell tw=0
