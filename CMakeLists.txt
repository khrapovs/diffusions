cmake_minimum_required(VERSION 3.15)
project(affidiff LANGUAGES C)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Find NumPy
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
  OUTPUT_VARIABLE NumPy_INCLUDE_DIRS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Cythonize the .pyx file
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m cython "src/affidiff/simulate.pyx"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  RESULT_VARIABLE CYTHON_RESULT
)

if(NOT CYTHON_RESULT EQUAL 0)
  message(FATAL_ERROR "Cythonization of simulate.pyx failed")
endif()

# Create the extension module from generated .c file
python_add_library(simulate MODULE
  "src/affidiff/simulate.c"
  WITH_SOABI
)

target_link_libraries(simulate PRIVATE)
target_include_directories(simulate PRIVATE ${NumPy_INCLUDE_DIRS})

# Install the module
install(TARGETS simulate DESTINATION affidiff)
