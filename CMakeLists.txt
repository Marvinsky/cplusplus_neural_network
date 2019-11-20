cmake_minimum_required(VERSION 3.7)


set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++17 -pthread")

project(NeuralNetwork)

file(GLOB SOURCES "src/*.cpp")

add_executable(NeuralNetwork ${SOURCES})
