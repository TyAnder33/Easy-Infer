cmake_minimum_required(VERSION 3.24)

project(EasyInfer VERSION 0.1.0 LANGUAGES CXX)

include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

add_library(easyinfer_core
    src/gpt2.cpp
    src/model_config.cpp
    src/model_registry.cpp
    src/weight_store.cpp
)
target_include_directories(easyinfer_core PUBLIC include)
target_compile_features(easyinfer_core PUBLIC cxx_std_20)
target_link_libraries(easyinfer_core PRIVATE nlohmann_json::nlohmann_json)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(easyinfer_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()

add_executable(engine src/main.cpp)
target_link_libraries(engine PRIVATE easyinfer_core)

include(CTest)
if(BUILD_TESTING)
    add_executable(model_config_test tests/model_config_test.cpp)
    target_link_libraries(model_config_test PRIVATE easyinfer_core)
    add_test(NAME model_config_test COMMAND model_config_test)
endif()
