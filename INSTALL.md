# Installation instructions Ubuntu
Installation is based on [KPConv](https://github.com/HuguesTHOMAS/KPConv-PyTorch) for PyTorch. The implementation has been tested on Ubuntu 18.04 with Python 3.7.11:

<ol>
<li>Clone the repository to any directory and create a new conda environment. You can use the <code>WeaSAL_CondaEnvList.txt</code> file to automatically install all dependencies and create a ready to use WeaSAL conda environment (in this case continue with step 6).</li>
<li>Install <a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html">CUDA</a>  and <a href="https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html">cuDNN</a> with the following configuration: <br>
    PyTorch 1.4.0, CUDA 10.1 and cuDNN 7.6</li>
<li>Ensure all python packages are installed:<br>
<code>
    sudo apt update 
</code><br>
<code>
    sudo apt install python3-dev python3-pip python3-tk
</code></li>
<li>Follow <a href="https://pytorch.org/get-started/locally/">PyTorch installation procedure</a> for version 1.4.0.</li>
<li>Install the other dependencies with conda:
<ul>
    <li>numpy</li>
    <li>scikit-learn</li>
    <li>PyYAML</li>
    <li>matplotlib (for visualization)</li>
    <li>mayavi (for visualization)</li>
    <li>PyQt5 (for visualization)</li>
</ul></li>
<li>Compile the C++ extension modules for python located in <code>/cpp_wrappers</code>. Open a terminal in this folder, and run:<br>
<code>
    sh compile_wrappers.sh
</code>
<li>Continue with setting up the point cloud data folder as described in <code>README.md</code></li>
</ol>


(Program was not tested on Windows)

