#!/bin/bash
[ -z $1 ] && { echo "Usage: $0 N , where N is the hw number in [1..6]" ; exit 1 ; }
rm -fr build
mkdir build
cd build
HW="HW$1" cmake ../
make

