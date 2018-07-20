import os
import sys
import re
import shutil
import filecmp, argparse 

def get_hip_file_path(filepath):
    """ Returns the new name of the hipified file """
    dirpath, filename = os.path.split(filepath)
    filename_without_ext, ext = os.path.splitext(filename)
    if "cudnn" in filename:
        hip_name = re.sub(r'cudnn', r'miopen', filename)
        if filename == "pool_op_cudnn.cu":
            hip_name = "pool_op_miopen.cc"
        if filename == "cudnn_wrappers.h":
            hip_name = "miopen_wrapper.h"
    elif "gpu" in filename_without_ext:
        hip_name = re.sub(r'gpu','hip',filename_without_ext)
        if ext == ".h":
            hip_name = hip_name + ext
        else:
            hip_name = hip_name + ".cc"
    else:
        if ext in [".cc", ".h"]:
            return filepath
        hip_name = filename_without_ext + "_hip.cc"
 
    hip_file_path = os.path.join(dirpath,hip_name)
    return hip_file_path

def get_hip_files(proj_dir):
    # get list of all hip files
    hip_files = set([]) 
    for dirname, _dirs, files in os.walk(proj_dir):
        if os.path.basename(dirname) == "hip":
            hip_files.update(files)
    return hip_files    

def get_cuda_files(proj_dir, hip_files): 
    cuda_files = dict()
    for dirname, _dirs, files in os.walk(proj_dir):
        base_dir = os.path.basename(dirname)
        if not base_dir == "hip":
            for file in files:
                if "cudnn" in file or "gpu" in file or file.endswith(".cu"):
                    hip_name = get_hip_file_path(os.path.join(dirname, file))
                    if os.path.basename(hip_name) in hip_files:
                        cuda_files[os.path.join(base_dir, file)] = os.path.join(dirname, file)
    return cuda_files

def compare_trees(new_files, old_files):
    mismatched_files = []
    matched_files = []
    missing_files_old = []
    missing_files_new = []
    for file in new_files:
        if file in old_files:
            if filecmp.cmp(new_files[file], old_files[file]):
                matched_files.append(file)
            else:
                mismatched_files.append(file)
        else:
            missing_files_old.append(file)

    for file in old_files:
        if file not in new_files:
            missing_files_new.append(file) 


    return mismatched_files, matched_files, missing_files_new, missing_files_old

def main():
    """
    parser = argparse.ArgumentParser(
        description="caffe2 upstream tracker")

    parser.add_argument('--work_dir',
                        type = str,
                        help = "working directory where tree's to be compared are present",
                        required = True)

    args = parser.parse_args()
    """
    # change dir to work_dir
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    old_tree_path = os.path.join(os.getcwd(), "pytorch_old")
    new_tree_path = os.path.join(os.getcwd(), "pytorch")
    if not os.path.exists(old_tree_path):
        print("ERROR: Old tree does not exist to compare")
        os.system("git clone --recursive https://github.com/pytorch/pytorch.git")
        shutil.copytree(new_tree_path, old_tree_path)
        shutil.rmtree(new_tree_path)
        sys.exit(1)

    # clone latest tree
    os.system("git clone --recursive https://github.com/pytorch/pytorch.git")

    hip_files_old = get_hip_files(old_tree_path)
    cuda_files_old = get_cuda_files(old_tree_path, hip_files_old)
    hip_files_new = get_hip_files(new_tree_path)
    cuda_files_new = get_cuda_files(new_tree_path, hip_files_new)
    mismatched_files, matched_files, missing_files_new, missing_files_old = compare_trees(cuda_files_new, cuda_files_old)

    print "=========== mismatched_files =========== \n", mismatched_files
    print "=========== missing_files_new =========== \n", missing_files_new
    print "=========== missing_files_old =========== \n", missing_files_old
    print "=========== matched_files =========== \n", matched_files

    shutil.rmtree(old_tree_path)
    shutil.copytree(new_tree_path, old_tree_path)
    shutil.rmtree(new_tree_path)

    if mismatched_files or missing_files_old or missing_files_new:
        print "Something has chamged"
        sys.exit(1)
        
if __name__ == '__main__':
    main()
