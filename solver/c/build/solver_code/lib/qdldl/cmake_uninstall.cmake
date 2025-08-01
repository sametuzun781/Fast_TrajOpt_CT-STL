if(NOT EXISTS "/Users/su781/Desktop/SCP_TrajGen/solver/c/build/install_manifest.txt")
  message(FATAL_ERROR "Cannot find install manifest: /Users/su781/Desktop/SCP_TrajGen/solver/c/build/install_manifest.txt")
endif(NOT EXISTS "/Users/su781/Desktop/SCP_TrajGen/solver/c/build/install_manifest.txt")

file(READ "/Users/su781/Desktop/SCP_TrajGen/solver/c/build/install_manifest.txt" files)
string(REGEX REPLACE "\n" ";" files "${files}")
foreach(file ${files})
  message(STATUS "Uninstalling $ENV{DESTDIR}${file}")
  if(IS_SYMLINK "$ENV{DESTDIR}${file}" OR EXISTS "$ENV{DESTDIR}${file}")
    exec_program(
      "/Users/su781/miniforge3/envs/trajopt/lib/python3.13/site-packages/cmake/data/bin/cmake" ARGS "-E remove \"$ENV{DESTDIR}${file}\""
      OUTPUT_VARIABLE rm_out
      RETURN_VALUE rm_retval
      )
    if(NOT "${rm_retval}" STREQUAL 0)
      message(FATAL_ERROR "Problem when removing $ENV{DESTDIR}${file}")
    endif(NOT "${rm_retval}" STREQUAL 0)
  else(IS_SYMLINK "$ENV{DESTDIR}${file}" OR EXISTS "$ENV{DESTDIR}${file}")
    message(STATUS "File $ENV{DESTDIR}${file} does not exist.")
  endif(IS_SYMLINK "$ENV{DESTDIR}${file}" OR EXISTS "$ENV{DESTDIR}${file}")
endforeach(file)
