# Evolution Gym (a Fork)
This is the official repository for the *GECCO'23* paper

**<a href="https://dl.acm.org/doi/abs/10.1145/3583131.3590429">How the Morphology Encoding Influences the Learning Ability in Body-Brain Co-Optimization</a>**
<br>
<a href="https://pigozzif.github.io">Federico Pigozzi</a>, Federico Camerota, and Eric Medvet
<br>

<div align="center">
<img src="teaser.gif"></img>
</div>

# Installation

Clone the repo and submodules:

```shell
git clone --recurse-submodules https://github.com/federico-camerota/evogym.git
```
and checkout to target branch:
```
git checkout baldwin
```
Please note that only the **baldwin** branch contains the experiments for this paper (not, for example, the main).

### Requirements

* Python 3.7/3.8
* Linux, macOS, or Windows with [Visual Studios 2017](https://visualstudio.microsoft.com/vs/older-downloads/)
* [OpenGL](https://www.opengl.org//)
* [CMake](https://cmake.org/download/)
* [PyTorch](http://pytorch.org/)

<!--- (See [installation instructions](#opengl-installation-on-unix-based-systems) on Unix based systems) --->

On **Linux only**:

```shell
sudo apt-get install xorg-dev libglu1-mesa-dev
```

Either install Python dependencies with conda:

```shell
conda env create -f environment.yml
conda activate evogym
```

or with pip:

```shell
pip install -r requirements.txt
```

### Build and Install Package

To build the C++ simulation, build all the submodules, and install `evogym` run the following command:

```shell
python setup.py install
``` 

### Test Installation

cd to the `examples` folder and run the following script:

```shell
python gym_test.py
```

This script creates a random `5x5` robot in the `Walking-v0` environment. The robot is taking random actions. A window should open with a visualization of the environment -- kill the process from the terminal to close it.

<!--### OpenGL installation on Unix-based systems

To install OpenGL via [homebrew](https://brew.sh/), run the following commands:

```shell
brew install glfw
```
--->

## Usage
Run:
```
python examples/run_ga.py
```
for the experiments with GA and run:
```
python examples/run_cppn_neat.py
```
for the experiments with CPPN-NEAT.

## Bibliography
Please cite as:
```
@inproceedings{pigozzi2023evorl,
  title={How the Morphology Encoding Influences the Learning Ability in Body-Brain Co-Optimization},
  author={Pigozzi, Federico and Camerota Verd{\`u}, Federico Julian and Medvet, Eric},
  booktitle={Proceedings of the Genetic and Evolutionary Computation Conference},
  pages={1045--1054},
  year={2023}
}
```
