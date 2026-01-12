#!/bin/bash

module load conda

# conda remove -y -n acl_ssl --all

for ((i=1; i<=27; i++)); do
    conda remove -y -n acl_ssl$i --all
done