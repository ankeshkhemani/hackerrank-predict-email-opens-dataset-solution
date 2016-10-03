#!/bin/bash

rm data/accurate_input_data.mat data/accurate_input_data.mat
rm data/accurate_input_data.mat data/accurate_test_data.mat

#Use a java function to Convert train and test csv files to mat files

java WriteMat data/accurate_input_data.csv data/accurate_input_data.mat
java WriteMat data/accurate_test_data.csv data/accurate_test_data.mat
./wrapper.m


