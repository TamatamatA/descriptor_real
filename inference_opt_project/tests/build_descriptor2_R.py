#!/usr/bin/env python3
"""
Build descriptor_all_R.cpp using pybind11 and g++
"""

import os
import subprocess
import sys
import sysconfig

def build_descriptor_with_gcc():
    """Build descriptor_all_R.cpp using g++ directly"""
    
    print("[*] Building descriptor_all_R.cpp using g++...")
    
    # Get Python info
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    include_path = sysconfig.get_path('include')
    
    # Get pybind11 include
    try:
        import pybind11
        pybind11_include = pybind11.get_include()
    except ImportError:
        print("[ERROR] pybind11 not found. Install: pip install pybind11")
        return False
    
    # Get Eigen include (if available)
    import importlib.util
    eigen_spec = importlib.util.find_spec("eigen")
    if eigen_spec and eigen_spec.origin:
        eigen_include = os.path.dirname(eigen_spec.origin)
    else:
        eigen_include = "/usr/include/eigen3"  # fallback
    
    # Get extension suffix
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if not ext_suffix:
        ext_suffix = f".cpython-{sys.version_info.major}{sys.version_info.minor}-x86_64-linux-gnu.so"
    
    output_file = f"descriptor_all2_R{ext_suffix}"
    source_file = "descriptor_all2_R.cpp"
    
    # Build command
    cmd = [
        "g++",
        "-O3",
        "-march=native",
        "-std=c++17",
        "-fopenmp",
        "-shared",
        "-fPIC",
        "-static-libstdc++",
        "-static-libgcc",
        f"-I{include_path}",
        f"-I{pybind11_include}",
        f"-I{eigen_include}",
        source_file,
        "-o", output_file,
        "-fopenmp"
    ]
    
    print(f"  Output: {output_file}")
    print(f"  Include paths:")
    print(f"    - Python: {include_path}")
    print(f"    - pybind11: {pybind11_include}")
    print(f"    - Eigen: {eigen_include}")
    print()
    print(f"  Command: {' '.join(cmd)}")
    print()
    
    # Run build
    try:
        result = subprocess.run(cmd, check=True, cwd=os.getcwd())
        
        if os.path.exists(output_file):
            print(f"[+] Build successful: {output_file}")
            print(f"    Size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
            return True
        else:
            print(f"[-] Output file not created: {output_file}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"[-] Build failed with error code {e.returncode}")
        print(f"    {e}")
        return False
    except Exception as e:
        print(f"[-] Build error: {e}")
        return False

if __name__ == "__main__":
    os.chdir('/mnt/c/Users/stama/Downloads/polymlp')
    
    success = build_descriptor_with_gcc()
    sys.exit(0 if success else 1)
