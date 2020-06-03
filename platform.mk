# Copyright (c) 2019, University of Washington All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
# Redistributions of source code must retain the above copyright notice, this list
# of conditions and the following disclaimer.
# 
# Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
# 
# Neither the name of the copyright holder nor the names of its contributors may
# be used to endorse or promote products derived from this software without
# specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# BSG_F1_DIR: The path to the BSG F1 repository
ifndef BSG_F1_DIR
$(error $(shell echo -e "$(RED)BSG MAKE ERROR: BSG_F1_DIR is not defined$(NC)"))
endif

# BSG_PLATFORM defines the platform to run or simulate on while
# running examples/regression. Current options are the directories in
# platforms of this repository

# We default to synopsys VCS-MX, HOWEVER, if VCS_HOME is not defined
# then we will assume that VCS/VCS-MX is not installed and try
# fall-back options
BSG_PLATFORM ?= vcs

# FIRST check if the user has defined BSG_PLATFORM externally
AVAILABLE_PLATFORMS=$(subst platforms/,,$(wildcard platforms/*) platforms/)

ifneq ($(filter $(BSG_PLATFORM),$(AVAILABLE_PLATFORMS)),)
    $(info $(BSG_PLATFORM) exists in $(AVAILABLE_PLATFORMS))
else
    $(error $(shell echo -e "$(RED)BSG MAKE ERROR: BSG_PLATFORM value $(BSG_PLATFORM) is not a valid platform$(NC)"))
endif

# TODO: What happens if VCS_HOME is not defined, and BSG_PLATFORM is overriden

ifdef VCS_HOME
# If VCS installed, continue as normal
else ifdef AGFI
# If VCS is not installed, check for AWS. We define AGFI in the
# environment of every AGFI we build.
BSG_PLATFORM := aws
else
$(error $(shell echo -e "$(RED)BSG MAKE ERROR: No working platforms available platform for $(NC)"))
endif



# BSG Platform Path is the path to the target platform, i.e. the
# substrate that actually runs the machine. 

# To switch platforms, simply switch the path of BSG_PLATFORM_PATH to
# another directory with a platform.mk file.
BSG_PLATFORM_PATH ?= $(BSG_F1_DIR)/platforms/$(BSG_PLATFORM)

# Convert the machine path to an abspath
override BSG_MACHINE_PATH := $(abspath $(BSG_MACHINE_PATH))
