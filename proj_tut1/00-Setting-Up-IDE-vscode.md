# 00-Setting Up IDE for Developing 

In this tutorial you will learn:

* How to install and setup vscode
* How to create virtual environment for vscode
* How to install the packages for this project


## 1. Download and Install Visual Studio Code (VS code)

Visual Studio Code (VS code) is one of the most popular lightweight IDE (Integrated Development Enviroment) for programming languages including python. Here we show how to download and install pycharm: 

- Go to official webiste [link](https://code.visualstudio.com/), download and install.
	 
<br/>

## 2. VScode

<br/>

### 2.1 Download codes

Dowload the tutorial codes from blackboard system. The zip file is called:

ProjectTutorial1.zip

unzip the items.
<br/>

### 2.2 Open codes as vscode project

Click `File`->`Open Folder` button on the navigation bar in vscode, and navigate to the root path your downloaded codes, click `open`.

If a diaglog pop up to ask you if you trust the author, just type yes.

</br>

### 2.3 how to open a terminal in vscode

Go to navigator bar on the top, type `Terminal`, type `New Terminal`.

</br>

## 3  Anaconda
</br>

### 3.1 Install Anaconda

Anaconda is a great package manager for python and can create virtual environments for different projects. 

Go to official website to download and install anaconda: 

- [https://www.anaconda.com/](https://www.anaconda.com/).

<br/>

It will take a long time to download.

</br>

### 3.2 create a virtual environemnt
 
Since python projects might rely on dependency specific to their project such as python version, package dependency, it is important to use virtual environment to manage these dependency. Conda provides the powerful virtual environment for python. 

In vscode, fisrt select conda interpreter. In the left right corner, select the interpreter button, and then select the python interpreter call ` base: conda`. here are the instruction picture:

<p align="center">
<img src="media/vs-interpreter.png" alt="drawing" width="600"/>
</p>

Open a terminal like previous section: Go to navigator bar on the top, type `Terminal`, type `New Terminal`.

And type in terminal to create a virtual environment for python

```sh
conda create -n torch python=3.6 anaconda

```

After finishing, type

```sh
conda activate torch
```

Then you finish creating the virtual environment. You will see there is a `(torch)` at the beginning of terminal line, which means that your virtual environment has bee activated.

To make vscode recognize the conda virtual environment, you can setup the python interpreter like previous section:
 
> In the left right corner, select the interpreter button, and then select the python interpreter call ` torch: conda`. here are the instruction picture: 

<p align="center">
<img src="media/vs-interpreter.png" alt="drawing" width="600"/>
</p>

> Note: 
> 1. make sure to select the virutal environment name called `torch`.
> 2. If you cannot find the desired environment name, type refresh button to update the settings.

The next time you open a new terminal, you will find conda virtual environment have been activated automatically. And it is also make preparation for the python debugger.

<br/>

### 3.3 Install packages

Install pytorch, got the official website of pytorch, find the command line corresponding to your machine:

<p align="center">
<img src="media/pytorch_install.png" alt="drawing" width="600"/>
</p>

> Tips: 
>  1. Since you have install ananconda, you can select `conda` to install
> 2. If you have a discrete GPU (such as RTX2080), you can just select CUDA to install. (If you have RTX2xxx series GPU, you can select CUDA10 or CUDA11. However, if you have the latest version of GPU, RTX3xxx series, then you only can choose CUDA11)
> 3. Macbook do not have discrete GPU, so do not need to install CUDA. If you do not have discrete GPU, You can also finish this project since this project is mainly running computation on CPU.

After you find the corresponding  command line, type in your terminal, for example:

```sh
<use your own command line instead> #conda install pytorch torchvision torchaudio -c pytorch
```

To install the rest of the dependency of our project, type

```sh
pip install -r requirements.txt
```

</br>


### 3.4 Install Jupyter Notebook

Open a terminal, type

```sh
conda install -c conda-forge jupyterlab

```

Here we need to type the following command to make jupyeter notebook recognize our virtual environment:

```sh
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=torch
```

Then you finish setup the jupyter notebook.




</br></br>
Congratulation! You have finished setting up the environment.





 
