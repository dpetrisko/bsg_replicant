# Amazon FPGA Hardware Development Kit
#
# Copyright 2016 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Amazon Software License (the "License"). You may not use
# this file except in compliance with the License. A copy of the License is
# located at
#
#    http://aws.amazon.com/asl/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or
# implied. See the License for the specific language governing permissions and
# limitations under the License.

package require tar

## Do not edit $TOP
set TOP top_sp

#################################################
## Command-line Arguments
#################################################
set timestamp           [lindex $argv  0]
set strategy            [lindex $argv  1]
set hdk_version         [lindex $argv  2]
set shell_version       [lindex $argv  3]
set device_id           [lindex $argv  4]
set vendor_id           [lindex $argv  5]
set subsystem_id        [lindex $argv  6]
set subsystem_vendor_id [lindex $argv  7]
set clock_recipe_a      [lindex $argv  8]
set clock_recipe_b      [lindex $argv  9]
set clock_recipe_c      [lindex $argv 10]
set uram_option         [lindex $argv 11]
set notify_via_sns      [lindex $argv 12]
set design_name         [lindex $argv 13]
set final_dcp_name      ${timestamp}.SH_CL_routed.dcp
set manifest_name       ${timestamp}.manifest.txt
if {[string compare $design_name ""] == 0} {
    set tar_name ${timestamp}.Developer_CL.tar
} else {
    set tar_name ${design_name}.Developer_CL.tar
}
##################################################
## Flow control variables 
##################################################
set cl.synth   1
set implement  1

#################################################
## Generate CL_routed.dcp (Done by User)
#################################################
puts "AWS FPGA Scripts";
puts "Creating Design Checkpoint from Custom Logic source code";
puts "HDK Version:            $hdk_version";
puts "Shell Version:          $shell_version";
puts "Vivado Script Name:     $argv0";
puts "Strategy:               $strategy";
puts "PCI Device ID           $device_id";
set ::env(device_id) $device_id
puts "PCI Vendor ID           $vendor_id";
set ::env(vendor_id) $vendor_id
puts "PCI Subsystem ID        $subsystem_id";
set ::env(subsystem_id) $subsystem_id
puts "PCI Subsystem Vendor ID $subsystem_vendor_id";
set ::env(subsystem_vendor_id) $subsystem_vendor_id
puts "Clock Recipe A:         $clock_recipe_a";
set ::env(CLOCK_A_RECIPE) [string index $clock_recipe_a end]
puts "Clock Recipe B:         $clock_recipe_b";
set ::env(CLOCK_B_RECIPE) [string index $clock_recipe_b end]
puts "Clock Recipe C:         $clock_recipe_c";
set ::env(CLOCK_C_RECIPE) [string index $clock_recipe_c end]
puts "URAM option:            $uram_option";
puts "Notify when done:       $notify_via_sns";

#checking if CL_DIR env variable exists
if { [info exists ::env(CL_DIR)] } {
        set CL_DIR $::env(CL_DIR)
        puts "Using CL directory $CL_DIR";
} else {
        puts "Error: CL_DIR environment variable not defined ! ";
        puts "Use export CL_DIR=Your_Design_Root_Directory"
        exit 2
}

#checking if HDK_SHELL_DIR env variable exists
if { [info exists ::env(HDK_SHELL_DIR)] } {
        set HDK_SHELL_DIR $::env(HDK_SHELL_DIR)
        puts "Using Shell directory $HDK_SHELL_DIR";
} else {
        puts "Error: HDK_SHELL_DIR environment variable not defined ! ";
        puts "Run the hdk_setup.sh script from the root directory of aws-fpga";
        exit 2
}

#checking if HDK_SHELL_DESIGN_DIR env variable exists
if { [info exists ::env(HDK_SHELL_DESIGN_DIR)] } {
        set HDK_SHELL_DESIGN_DIR $::env(HDK_SHELL_DESIGN_DIR)
        puts "Using Shell design directory $HDK_SHELL_DESIGN_DIR";
} else {
        puts "Error: HDK_SHELL_DESIGN_DIR environment variable not defined ! ";
        puts "Run the hdk_setup.sh script from the root directory of aws-fpga";
        exit 2
}

##################################################
### Output Directories used by step_user.tcl
##################################################
set implDir   $CL_DIR/build/checkpoints
set rptDir    $CL_DIR/build/reports
set cacheDir  $HDK_SHELL_DESIGN_DIR/cache/ddr4_phy

puts "All reports and intermediate results will be time stamped with $timestamp";

set_msg_config -id {Chipscope 16-3} -suppress
set_msg_config -string {AXI_QUAD_SPI} -suppress

# Suppress Warnings
# These are to avoid warning messages that may not be real issues. A developer
# may comment them out if they wish to see more information from warning
# messages.
set_msg_config -id {Constraints 18-550} -suppress
set_msg_config -id {Constraints 18-619} -suppress
set_msg_config -id {DRC 23-20}          -suppress
set_msg_config -id {Physopt 32-742}     -suppress
set_msg_config -id {Place 46-14}        -suppress
set_msg_config -id {Synth 8-3295}       -suppress
set_msg_config -id {Synth 8-3321}       -suppress
set_msg_config -id {Synth 8-3331}       -suppress
set_msg_config -id {Synth 8-3332}       -suppress
set_msg_config -id {Synth 8-350}        -suppress
set_msg_config -id {Synth 8-3848}       -suppress
set_msg_config -id {Synth 8-3917}       -suppress
set_msg_config -id {Timing 38-436}      -suppress
set_msg_config -id {Synth 8-6014}       -suppress
set_msg_config -id {Constraints 18-952} -suppress
set_msg_config -id {DRC CKLD-2}         -suppress
set_msg_config -id {DRC REQP-1853}      -suppress
set_msg_config -id {Route 35-456}       -suppress
set_msg_config -id {Route 35-455}       -suppress
set_msg_config -id {Route 35-459}       -suppress

puts "AWS FPGA: ([clock format [clock seconds] -format %T]) Calling the encrypt.tcl.";

# Check that an email address has been set, else unset notify_via_sns

if {[string compare $notify_via_sns "1"] == 0} {
  if {![info exists env(EMAIL)]} {
    puts "AWS FPGA: ([clock format [clock seconds] -format %T]) EMAIL variable empty!  Completition notification will *not* be sent!";
    set notify_via_sns 0;
  } else {
    puts "AWS FPGA: ([clock format [clock seconds] -format %T]) EMAIL address for completion notification set to $env(EMAIL).";
  }
}

##################################################
### Strategy options 
##################################################
switch $strategy {
    "BASIC" {
        puts "BASIC strategy."
        source $HDK_SHELL_DIR/build/scripts/strategy_BASIC.tcl
    }
    "EXPLORE" {
        puts "EXPLORE strategy."
        source $HDK_SHELL_DIR/build/scripts/strategy_EXPLORE.tcl
    }
    "TIMING" {
        puts "TIMING strategy."
        source $HDK_SHELL_DIR/build/scripts/strategy_TIMING.tcl
    }
    "CONGESTION" {
        puts "CONGESTION strategy."
        source $HDK_SHELL_DIR/build/scripts/strategy_CONGESTION.tcl
    }
    "DEFAULT" {
        puts "DEFAULT strategy."
        source $HDK_SHELL_DIR/build/scripts/strategy_DEFAULT.tcl
    }
    default {
        puts "$strategy is NOT a valid strategy. Defaulting to strategy DEFAULT."
        source $HDK_SHELL_DIR/build/scripts/strategy_DEFAULT.tcl
    }
}

#Encrypt source code
set VIVADO_IP_DIR $::env(XILINX_VIVADO)/data/ip/xilinx/
set TARGET_DIR $CL_DIR/build/src_post_encryption
set UNUSED_TEMPLATES_DIR $HDK_SHELL_DIR/design/interfaces

# OK - here, and in synth.tcl is how we handle Xilinx IP in the
# HDL flow. Things get... a little weird. 
#
# The following lines copy the source files of any verilog and VHDL IP
# that we use in the design. It's ugly, but effective, and I can't
# figure out how to do it any better. In essence, for every IP that we
# use, we copy the HDL file from it's xilinx IP directory, into
# $TARGET_DIR. However, it's not that simple, because each IP can
# depend on underlying IP -- for example the AXI MM FIFO
# (axi_fifo_mm_s) uses the Fifo Library (lib_fifo), so you ALSO have
# to copy the Fifo Library. The only way you can determine what IP
# files are dependencies is by reading the component.xml file and
# finding the Verilog/VHDL Synthesis Fileset. Repeat (recursively)
# until no more IPs remain.
#
# As I said, it's janky.
#
# If you want to know how the files are used by vivado once they are
# copied, read synth.tcl.
file copy -force $VIVADO_IP_DIR/generic_baseblocks_v2_1/hdl/generic_baseblocks_v2_1_vl_rfs.v        $TARGET_DIR
file copy -force $VIVADO_IP_DIR/axi_register_slice_v2_1/hdl/axi_register_slice_v2_1_vl_rfs.v        $TARGET_DIR
file copy -force $VIVADO_IP_DIR/axi_crossbar_v2_1/hdl/axi_crossbar_v2_1_vl_rfs.v                    $TARGET_DIR
file copy -force $VIVADO_IP_DIR/axi_dwidth_converter_v2_1/hdl/axi_dwidth_converter_v2_1_vlsyn_rfs.v $TARGET_DIR
file copy -force $VIVADO_IP_DIR/axi_data_fifo_v2_1/hdl/axi_data_fifo_v2_1_vl_rfs.v                  $TARGET_DIR
file copy -force $VIVADO_IP_DIR/axi_infrastructure_v1_1/hdl/axi_infrastructure_v1_1_0.vh            $TARGET_DIR
file copy -force $VIVADO_IP_DIR/axi_infrastructure_v1_1/hdl/axi_infrastructure_v1_1_vl_rfs.v        $TARGET_DIR
file copy -force $VIVADO_IP_DIR/axi_lite_ipif_v3_0/hdl/axi_lite_ipif_v3_0_vh_rfs.vhd                $TARGET_DIR
file copy -force $VIVADO_IP_DIR/axi_fifo_mm_s_v4_1/hdl/axi_fifo_mm_s_v4_1_rfs.vhd                   $TARGET_DIR

file copy -force $VIVADO_IP_DIR/fifo_generator_v13_2/hdl/fifo_generator_v13_2_vhsyn_rfs.vhd         $TARGET_DIR
file copy -force $VIVADO_IP_DIR/blk_mem_gen_v8_4/hdl/blk_mem_gen_v8_4_vhsyn_rfs.vhd                 $TARGET_DIR
file copy -force $VIVADO_IP_DIR/lib_fifo_v1_0/hdl/lib_fifo_v1_0_rfs.vhd                             $TARGET_DIR
file copy -force $VIVADO_IP_DIR/lib_pkg_v1_0/hdl/lib_pkg_v1_0_rfs.vhd                               $TARGET_DIR
file copy -force $VIVADO_IP_DIR/axis_register_slice_v1_1/hdl/axis_register_slice_v1_1_vl_rfs.v      $TARGET_DIR
file copy -force $VIVADO_IP_DIR/axis_infrastructure_v1_1/hdl/axis_infrastructure_v1_1_0.vh          $TARGET_DIR
file copy -force $VIVADO_IP_DIR/axis_infrastructure_v1_1/hdl/axis_infrastructure_v1_1_vl_rfs.v      $TARGET_DIR
file copy -force $VIVADO_IP_DIR/axis_dwidth_converter_v1_1/hdl/axis_dwidth_converter_v1_1_vl_rfs.v  $TARGET_DIR

file copy -force $UNUSED_TEMPLATES_DIR/unused_apppf_irq_template.inc          $TARGET_DIR
file copy -force $UNUSED_TEMPLATES_DIR/unused_cl_sda_template.inc             $TARGET_DIR
file copy -force $UNUSED_TEMPLATES_DIR/unused_ddr_a_b_d_template.inc          $TARGET_DIR
file copy -force $UNUSED_TEMPLATES_DIR/unused_ddr_c_template.inc              $TARGET_DIR
file copy -force $UNUSED_TEMPLATES_DIR/unused_dma_pcis_template.inc           $TARGET_DIR
file copy -force $UNUSED_TEMPLATES_DIR/unused_pcim_template.inc               $TARGET_DIR
file copy -force $UNUSED_TEMPLATES_DIR/unused_sh_bar1_template.inc            $TARGET_DIR
file copy -force $UNUSED_TEMPLATES_DIR/unused_flr_template.inc                $TARGET_DIR

# Make sure files have write permissions for the encryption
exec chmod +w {*}[glob $TARGET_DIR/*]

# encrypt .v/.sv/.vh/inc as verilog files
encrypt -k $HDK_SHELL_DIR/build/scripts/vivado_keyfile.txt -lang verilog  [glob -nocomplain -- $TARGET_DIR/*.?v] [glob -nocomplain -- $TARGET_DIR/*.vh] [glob -nocomplain -- $TARGET_DIR/*.inc]

# encrypt *vhdl files
encrypt -k $HDK_SHELL_DIR/build/scripts/vivado_vhdl_keyfile.txt -lang vhdl -quiet [ glob -nocomplain -- $TARGET_DIR/*.vhd? ]

# source encrypt.tcl

#Set the Device Type
source $HDK_SHELL_DIR/build/scripts/device_type.tcl

#Procedure for running various implementation steps (impl_step)
source $HDK_SHELL_DIR/build/scripts/step_user.tcl -notrace

########################################
## Generate clocks based on Recipe 
########################################

puts "AWS FPGA: ([clock format [clock seconds] -format %T]) Calling aws_gen_clk_constraints.tcl to generate clock constraints from developer's specified recipe.";

source $HDK_SHELL_DIR/build/scripts/aws_gen_clk_constraints.tcl

##################################################
### CL XPR OOC Synthesis
##################################################
if {${cl.synth}} {
   source -notrace ./synth.tcl
}

##################################################
### Implementation
##################################################
if {$implement} {

   ########################
   # Link Design
   ########################
   if {$link} {
      ####Create in-memory prjoect and setup IP cache location
      create_project -part [DEVICE_TYPE] -in_memory
      set_property IP_REPO_PATHS $cacheDir [current_project]
      puts "\nAWS FPGA: ([clock format [clock seconds] -format %T]) - Combining Shell and CL design checkpoints";
      add_files $HDK_SHELL_DIR/build/checkpoints/from_aws/SH_CL_BB_routed.dcp
      add_files $CL_DIR/build/checkpoints/${timestamp}.CL.post_synth.dcp
      set_property SCOPED_TO_CELLS {WRAPPER_INST/CL} [get_files $CL_DIR/build/checkpoints/${timestamp}.CL.post_synth.dcp]

      #Read the constraints, note *DO NOT* read cl_clocks_aws (clocks originating from AWS shell)
      read_xdc [ list \
         $CL_DIR/build/constraints/cl_pnr_user.xdc
      ]
      set_property PROCESSING_ORDER late [get_files cl_pnr_user.xdc]

      puts "\nAWS FPGA: ([clock format [clock seconds] -format %T]) - Running link_design";
      link_design -top $TOP -part [DEVICE_TYPE] -reconfig_partitions {WRAPPER_INST/SH WRAPPER_INST/CL}

      puts "\nAWS FPGA: ([clock format [clock seconds] -format %T]) - PLATFORM.IMPL==[get_property PLATFORM.IMPL [current_design]]";
      ##################################################
      # Apply Clock Properties for Clock Table Recipes
      ##################################################
      puts "AWS FPGA: ([clock format [clock seconds] -format %T]) - Sourcing aws_clock_properties.tcl to apply properties to clocks. ";
      
      # Apply properties to clocks
      source $HDK_SHELL_DIR/build/scripts/aws_clock_properties.tcl

      # Write post-link checkpoint
      puts "\nAWS FPGA: ([clock format [clock seconds] -format %T]) - Writing post-link_design checkpoint ${timestamp}.post_link.dcp";
      write_checkpoint -force $CL_DIR/build/checkpoints/${timestamp}.post_link.dcp
   }

   ########################
   # CL Optimize
   ########################
   if {$opt} {
      puts "\nAWS FPGA: ([clock format [clock seconds] -format %T]) - Running optimization";
      impl_step opt_design $TOP $opt_options $opt_directive $opt_preHookTcl $opt_postHookTcl
      if {$psip} {
         impl_step opt_design $TOP "-merge_equivalent_drivers -sweep"
      }
   }

# Constraints for TCK<->Main Clock
#set_clock_groups -name tck_clk_main_a0 -asynchronous -group [get_clocks -of_objects [get_pins static_sh/SH_DEBUG_BRIDGE/inst/bsip/inst/USE_SOFTBSCAN.U_TAP_TCKBUFG/O]] -group [get_clocks -of_objects [get_pins SH/kernel_clks_i/clkwiz_sys_clk/inst/CLK_CORE_DRP_I/clk_inst/mmcme3_adv_inst/CLKOUT0]]
#set_clock_groups -name tck_drck -asynchronous -group [get_clocks -of_objects [get_pins static_sh/SH_DEBUG_BRIDGE/inst/bsip/inst/USE_SOFTBSCAN.U_TAP_TCKBUFG/O]] -group [get_clocks drck]
#set_clock_groups -name tck_userclk -asynchronous -group [get_clocks -of_objects [get_pins static_sh/SH_DEBUG_BRIDGE/inst/bsip/inst/USE_SOFTBSCAN.U_TAP_TCKBUFG/O]] -group [get_clocks -of_objects [get_pins static_sh/pcie_inst/inst/gt_top_i/diablo_gt.diablo_gt_phy_wrapper/phy_clk_i/bufg_gt_userclk/O]]


   ########################
   # CL Place
   ########################
   if {$place} {
      puts "\nAWS FPGA: ([clock format [clock seconds] -format %T]) - Running placement";
      if {$psip} {
         append place_options " -fanout_opt"
      }
      impl_step place_design $TOP $place_options $place_directive $place_preHookTcl $place_postHookTcl
   }

   ##############################
   # CL Post-Place Optimization
   ##############################
   if {$phys_opt} {
      puts "\nAWS FPGA: ([clock format [clock seconds] -format %T]) - Running post-place optimization";
      impl_step phys_opt_design $TOP $phys_options $phys_directive $phys_preHookTcl $phys_postHookTcl
   }

   ########################
   # CL Route
   ########################
   if {$route} {
      puts "\nAWS FPGA: ([clock format [clock seconds] -format %T]) - Routing design";
      impl_step route_design $TOP $route_options $route_directive $route_preHookTcl $route_postHookTcl
   }

   ##############################
   # CL Post-Route Optimization
   ##############################
   set SLACK [get_property SLACK [get_timing_paths]]
   #Post-route phys_opt will not be run if slack is positive or greater than -200ps.
   if {$route_phys_opt && $SLACK > -0.400 && $SLACK < 0} {
      puts "\nAWS FPGA: ([clock format [clock seconds] -format %T]) - Running post-route optimization";
      impl_step route_phys_opt_design $TOP $post_phys_options $post_phys_directive $post_phys_preHookTcl $post_phys_postHookTcl
   }

   ##############################
   # Final Implmentation Steps
   ##############################
   # Report final timing
   report_timing_summary -file $CL_DIR/build/reports/${timestamp}.SH_CL_final_timing_summary.rpt

   # This is what will deliver to AWS
   puts "AWS FPGA: ([clock format [clock seconds] -format %T]) - Writing final DCP to to_aws directory.";

   write_checkpoint -force $CL_DIR/build/checkpoints/to_aws/${final_dcp_name}
   
   # Generate debug probes file. Uncomment line below if debugging is enabled 
   write_debug_probes -force -no_partial_ltxfile -file $CL_DIR/build/checkpoints/${timestamp}.debug_probes.ltx

   close_project
}

# ################################################
# Create Manifest and Tarball for delivery
# ################################################

# Create a zipped tar file, that would be used for createFpgaImage EC2 API

puts "AWS FPGA: ([clock format [clock seconds] -format %T]) - Compress files for sending to AWS. "

# Create manifest file
set manifest_file [open "$CL_DIR/build/checkpoints/to_aws/${manifest_name}" w]
set hash [lindex [split [exec sha256sum $CL_DIR/build/checkpoints/to_aws/$final_dcp_name] ] 0]
set vivado_version [string range [version -short] 0 5]
puts "vivado_version is $vivado_version\n"

puts $manifest_file "manifest_format_version=2\n"
puts $manifest_file "pci_vendor_id=$vendor_id\n"
puts $manifest_file "pci_device_id=$device_id\n"
puts $manifest_file "pci_subsystem_id=$subsystem_id\n"
puts $manifest_file "pci_subsystem_vendor_id=$subsystem_vendor_id\n"
puts $manifest_file "dcp_hash=$hash\n"
puts $manifest_file "shell_version=$shell_version\n"
puts $manifest_file "tool_version=v$vivado_version\n"
puts $manifest_file "dcp_file_name=${final_dcp_name}\n"
puts $manifest_file "hdk_version=$hdk_version\n"
puts $manifest_file "date=$timestamp\n"
puts $manifest_file "clock_recipe_a=$clock_recipe_a\n"
puts $manifest_file "clock_recipe_b=$clock_recipe_b\n"
puts $manifest_file "clock_recipe_c=$clock_recipe_c\n"

close $manifest_file

# Delete old tar file with same name
if { [file exists $CL_DIR/build/checkpoints/to_aws/${tar_name}] } {
    puts "Deleting old tar file with same name.";
    file delete -force $CL_DIR/build/checkpoints/to_aws/${tar_name}
}

# Tar checkpoint to aws
cd $CL_DIR/build/checkpoints/to_aws/
tar::create ${tar_name} [ list ${manifest_name} ${final_dcp_name} ]

puts "AWS FPGA: ([clock format [clock seconds] -format %T]) - Finished creating final tar file in to_aws directory.";

if {[string compare $notify_via_sns "1"] == 0} {
  puts "AWS FPGA: ([clock format [clock seconds] -format %T]) - Calling notification script to send e-mail to $env(EMAIL)";
  exec $env(AWS_FPGA_REPO_DIR)/shared/bin/scripts/notify_via_sns.py
}

puts "AWS FPGA: ([clock format [clock seconds] -format %T]) - Build complete.";


