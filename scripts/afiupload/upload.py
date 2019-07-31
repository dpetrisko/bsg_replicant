#!/usr/bin/python3
import os
import git
import boto3
import datetime
import argparse 
import json
from TarAction import TarAction
from NameAction import NameAction
from BucketAction import BucketAction
from RepoAction import RepoAction
from PathAction import PathAction
from VersionAction import VersionAction

def setup_argparse():
    parser = argparse.ArgumentParser(description='Build an AWS FPGA Image (AFI)' + 
                                     'and upload it to the cloud')
    parser.add_argument("BuildDirectory", action=PathAction, nargs=1,
                        help='Path to Git Repositories')
    parser.add_argument('ImageName', action=NameAction, nargs=1,
                        help='The base name of the FPGA Image')
    parser.add_argument('ImageVersion', action=VersionAction, nargs=1,
                        help='Version number of the FPGA Image')
    parser.add_argument('TarPath', action=TarAction, nargs=1,
                        help='Path to a .tar file generated by the AWS build flow')
    parser.add_argument('BucketName', action=BucketAction, nargs=1,
                        help='Name of the AWS bucket')
    parser.add_argument('Description', type=str, nargs=1,
                        help='A string describing the FPGA image.')
    parser.add_argument('-c', '--configuration', type=str, nargs=1, 
                        default=["Not-Provided"],
                        help='String describing the FPGA image configuration')
    parser.add_argument('-r', action=RepoAction, nargs='+',
                        help='A repository directory name in the Build Directory')
    parser.add_argument('-d', '--dryrun', action='store_const', const=True,default=False,
                        help='Upload the image, but do not process it (dry run)')
    return parser

def construct_name(name, version, config):
    return '{} v{} (Configuration: {})'.format(name, str(version), config)

def upload_tar(args):
    s3 = boto3.client('s3')

    tar_path = args.TarPath
    tar_file = os.path.basename(tar_path)

    bucket = args.BucketName
    ymd = timestamp = datetime.datetime.now().strftime('%Y%m%d')
    tar_key = os.path.join(args.ImageName, ymd, str(args.ImageVersion), tar_file)

    print("Uploading {} to s3://{}/{}".format(tar_file, bucket.name, tar_key))
    s3.upload_file(tar_path, bucket.name, tar_key)

    return (bucket, tar_key)

def process_tar(args):
    ec2 = boto3.client('ec2')
    region = 'us-west-2'

    (bucket, tar_key) = upload_tar(args)
    log_key = tar_key + '.log'

    name = construct_name(args.ImageName, args.ImageVersion, args.configuration[0])
    desc = args.Description[0]

    print("Processing {}".format(tar_key))
    rsp = ec2.create_fpga_image(
        InputStorageLocation={
            'Bucket': bucket.name,
            'Key': tar_key
        },
        LogsStorageLocation={
            'Bucket': bucket.name,
            'Key': log_key
        },
        Description=desc,
        Name=name,
        DryRun=args.dryrun
    )

    tags = [{'Key':n,'Value':h['commit']} for (n,h) in args.r.items()]
    tags.append({'Key':'Version','Value':args.ImageVersion})
    tags.append({'Key':'Project','Value':args.ImageName})
    tags.append({'Key':'Configuration','Value':args.configuration[0]})
    ec2.create_tags(Resources=[rsp['FpgaImageId']],Tags=tags)

    print("New AFI: {}".format(rsp['FpgaImageId']))
    print("New AGFI: {}".format(rsp['FpgaImageGlobalId']))
    print("AFI Name: {}".format(name))
    print("AFI Description: {}".format(desc))
    return rsp

rsp = process_tar(setup_argparse().parse_args())

with open('upload.json', 'w') as f:
    json.dump(rsp, f)
