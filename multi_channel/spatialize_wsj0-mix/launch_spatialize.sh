#!/bin/bash

# Copyright 2018 Mitsubishi Electric Research Labs
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

function dispNrun {
        echo "--------------------------------"
        echo "RUNNING:" $1
        echo "--------------------------------"
        eval $1 &
}

NUM_SPEAKERS=2
MIN_OR_MAX="'min'"
FS=8000
NUM_JOBS=20
NUM_FILES=28000
STEP=$(( ${NUM_FILES} / ${NUM_JOBS} ))
for (( i=1; i<=${NUM_FILES}; i+=${STEP})); do
    cmd="spatialize_wsj0_mix(${NUM_SPEAKERS},${MIN_OR_MAX},${FS},$i,$i+${STEP}-1)"
    dispNrun "echo \"$cmd\" | matlab -nodisplay"
done
