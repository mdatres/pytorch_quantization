#!/usr/bin/env bash

PROBLEM=${1} # TODO: check that this is alphanumeric

PATH_CFG=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
PATH_QUANTLAB_HOME=$(cd $(dirname ${PATH_CFG}) && pwd)

# retrieve (hard) systems folder
PATH_STORAGE_CFG=${PATH_CFG}/storage_cfg.json
PYTHON_READ_STORAGE_CFG_DATA="import sys; import json; fp = open('${PATH_STORAGE_CFG}', 'r'); d = json.load(fp); fp.close(); print(d['data'])"
PATH_HARD_SYSTEMS_PKG_4DATA=$(python -c "${PYTHON_READ_STORAGE_CFG_DATA}")/$(basename ${PATH_QUANTLAB_HOME})/experiments
STR='/pytorch_quantization/experiments'
if [ ! -d "${PATH_HARD_SYSTEMS_PKG_4DATA}" ]
then
   # set up problem package
   PATH_HARD_SYSTEMS_PKG_4DATA=${PATH_QUANTLAB_HOME}/experiments
    PATH_PROBLEM_PKG=${PATH_QUANTLAB_HOME}/experiments/${PROBLEM}
    if [ -d "${PATH_PROBLEM_PKG}" ] && [ -f "${PATH_PROBLEM_PKG}/__init__.py" ]
    then
        echo "[PyTorch-Quantization] It seems that the QuantLab package for problem ${PROBLEM} has already been created..."
    else
        mkdir ${PATH_PROBLEM_PKG}

    fi

    # create (hard) data folder
    PATH_HARD_PROBLEM_PKG_DATA=${PATH_HARD_SYSTEMS_PKG_4DATA}/${PROBLEM}/data
    PATH_HARD_PROBLEM_PKG_LOGS=${PATH_HARD_SYSTEMS_PKG_4DATA}/${PROBLEM}/logs
    touch ${PATH_PROBLEM_PKG}/config.json
    mkdir -p ${PATH_HARD_PROBLEM_PKG_DATA}
    mkdir -p ${PATH_HARD_PROBLEM_PKG_LOGS}
    mkdir -p ${PATH_HARD_PROBLEM_PKG_LOGS}/saves
    mkdir -p ${PATH_HARD_PROBLEM_PKG_LOGS}/tensorboard
    
    echo "[PyTorch-Quantization] Remember to prepare the data for problem ${PROBLEM} at <$(dirname ${PATH_HARD_PROBLEM_PKG_DATA})>."

    if [ ! "$PATH_HARD_SYSTEMS_PKG_4DATA"=="$STR" ]
    then
        # link (soft) data folder to (hard) data folder
        ln -s ${PATH_HARD_PROBLEM_PKG_DATA} ${PATH_PROBLEM_PKG}/data
    fi

else

    # set up problem package
    PATH_PROBLEM_PKG=${PATH_QUANTLAB_HOME}/experiments/${PROBLEM}
    if [ -d "${PATH_PROBLEM_PKG}" ] && [ -f "${PATH_PROBLEM_PKG}/__init__.py" ]
    then
        echo "[PyTorch-Quantization] It seems that the QuantLab package for problem ${PROBLEM} has already been created..."
    else
        mkdir ${PATH_PROBLEM_PKG}

    fi

    # create (hard) data folder
    PATH_HARD_PROBLEM_PKG_DATA=${PATH_HARD_SYSTEMS_PKG_4DATA}/${PROBLEM}/data
    PATH_HARD_PROBLEM_PKG_LOGS=${PATH_HARD_SYSTEMS_PKG_4DATA}/${PROBLEM}/logs
    touch ${PATH_PROBLEM_PKG}/config.json
    mkdir -p ${PATH_HARD_PROBLEM_PKG_DATA}
    mkdir -p ${PATH_HARD_PROBLEM_PKG_LOGS}
    mkdir -p ${PATH_HARD_PROBLEM_PKG_LOGS}/saves
    mkdir -p ${PATH_HARD_PROBLEM_PKG_LOGS}/tensorboard
    
    echo "[PyTorch-Quantization] Remember to prepare the data for problem ${PROBLEM} at <$(dirname ${PATH_HARD_PROBLEM_PKG_DATA})>."

    if [! "$PATH_HARD_SYSTEMS_PKG_4DATA"=="$STR"]
    then
        # link (soft) data folder to (hard) data folder
        ln -s ${PATH_HARD_PROBLEM_PKG_DATA} ${PATH_PROBLEM_PKG}/data
    fi

fi