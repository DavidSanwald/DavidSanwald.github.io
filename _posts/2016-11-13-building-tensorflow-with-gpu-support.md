---
layout: post
title: Getting CUDA 8 to Work With openAI Gym on AWS and Compiling Tensorflow for CUDA 8 Compatibility
---
The necessary steps to get CUDA and cuDNN to work with an virtual framebuffer like xvfb, so that you can use openAI Gym. Also included how to compile Tensorflow using Google's Bazel.
<!--more-->

I had some hard time getting Tensorflow with GPU support and [OpenAI Gym](https://gym.openai.com/) at the same time working on an AWS EC2 instance, and it seems like I'm in good [company](https://github.com/openai/gym/issues/247). For some time I used [NVIDIA-Docker](https://github.com/NVIDIA/nvidia-docker/) for this but as much as I love Docker, depending on special access to the (NVIDIA) GPU drivers, took away some of the biggest advantages when using Docker, at least for my use cases. Running OpenAI Gym in a normal container, exposing some port to the outside and running agents/neural nets etc. elsewhere seems like a really promising approach and I'm looking forward to it being ready.

There are good [explanations](https://alliseesolutions.wordpress.com/2016/09/08/install-gpu-tensorflow-from-sources-w-ubuntu-16-04-and-cuda-8-0-rc/) on how to get Tensorflow with CUDA going, those were pretty helpful to me. However I suppose they were mostly concerned with supervised learning.
If you want to run certain OpenAI Gym environments headless on a server, you have to provide an X-server to them, even when you don't want to render a video. You can use a virtual framebuffer like xvfb for this, it works fine. But I never could getting it to work with GLX support. Also other solutions like X-Dummy failed.
The problem is, that there's no way to keep NVIDIA from installing OpenGl-libs, when using packages from some repo (which most of the tutorials do, because it's way more convenient). Finally [this](https://github.com/openai/gym/issues/366#issuecomment-251967650) comment by pemami4911 in a github issue pointed me into the right direction. Since many people seem to run into the same problems, maybe the following will spare you some trouble.

I'm using Python 3 because it really annoys me, that everyone still uses Python 2.7 in the deep learning community. Also I'm using Ubuntu Ubuntu 16.04 LTS XENIAL XERUS because it's released for ages (even the official Canonical AMI on AWS) and when spinning up a single new instance for computing just a few things, I don't see why I would use Ubuntu 14.04, just because 16.04 is not among the three AMIs AWS offers to me first.
But I think it should be no trouble to adapt this to other needs.

OT: *I also recommend using the AWS CLI together with the [oh-my-zsh](https://github.com/robbyrussell/oh-my-zsh) AWS plugin, because zsh and especially oh-my-zsh is awesome.*

Okay, let's begin:

Go to  <https://cloud-images.ubuntu.com/locator/ec2/> and look for the Ubuntu 16.04 LST XENIAL XERUS hvm:ebs-ssd	AMI from Canonical. For eu-central-1 it's **ami-8504fdea**.
Spin up a new EC2 GPU instance (**g2.2xlarge** will do) using that AMI (use a spot instance if you are as broke as me). I recommend using at least 20GB as root volume.
SSH into your instance, username is **ubuntu**.



{% highlight bash %}
$ sudo apt-get update
$ sudo apt-get -y dist-upgrade
{% endhighlight %}
Let's get some basics:

{% highlight bash %}
$ sudo apt-get install openjdk-8-jdk git python-dev python3-dev python-numpy python3-numpy build-essential python-pip python3-pip python3-venv swig python3-wheel libcurl3-dev
{% endhighlight %}

The Java stuff is for [Bazel](https://bazel.build/) Google's build tool we will use later for compiling Tensorflow. We will use openJDK but if you have to, you can also use the proprietary one from Oracle.

Also  some kernel sources, compilers and other stuff:

{% highlight bash %}
$ sudo apt-get install -y gcc g++ gfortran  git linux-image-generic linux-headers-generic linux-source linux-image-extra-virtual libopenblas-dev
{% endhighlight %}

As we are on it, let's install Bazel right now:

{% highlight bash %}
$ echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
$ curl https://storage.googleapis.com/bazel-apt/doc/apt-key.pub.gpg | sudo apt-key add -
$ sudo apt-get update
$ sudo apt-get install bazel
$ sudo apt-get upgrade bazel
{% endhighlight %}


Now curl or wget the right NVIDIA driver. For the GRID K520 GPU **367.57** should be the right choice (maybe Linus would it [call](https://www.youtube.com/watch?v=iYWzMvlj2RQ) the least wrong choice at most).
{% highlight bash %}
$ wget -P ~/Downloads/ http://us.download.nvidia.com/XFree86/Linux-x86_64/367.57/NVIDIA-Linux-x86_64-367.57.run
{% endhighlight %}
NVIDIA will clash with the nouveau driver so deactivate it:
{% highlight bash %}
$ sudo nano /etc/modprobe.d/blacklist-nouveau.conf
{% endhighlight %}
Insert the following lines and save:
{% highlight bash%}
blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
alias nouveau off
alias lbm-nouveau off
{% endhighlight %}
Update the initframs (basically functionality to mount your real rootfs, which has been outsourced from the kernel) and reboot:
{% highlight bash %}
$ sudo update-initramfs -u
$ sudo reboot
{% endhighlight %}
Make the NVIDIA driver runfile executable and install the driver and reboot one more time, just to be sure.

**IMPORTANT: In my experience xvfb will only work if you use the --no-opengl-files option!**
{% highlight bash %}
$ chmod +x ~/Downloads/Linux-x86_64/367.57/NVIDIA-Linux-x86_64-367.57.run
$ sudo sh ~/Linux-x86_64/367.57/NVIDIA-Linux-x86_64-367.57.run --no-opengl-files
$ sudo reboot
{% endhighlight %}
Now wget CUDA 8.0. toolkid from NVIDIA
{% highlight bash %}h
$ wget https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_linux-run
{% endhighlight %}
While downloading register at NVIDIA, download CUDNN 5 runtinme lib on your local machine and SCP it to the remot instance:
{% highlight bash %}
$ scp cudnn-8.0-linux-x64-v5.1.tgz ubuntu@ec2-35-156-52-27.eu-central-1.compute.amazonaws.com:~/Downloads/
{% endhighlight %}
Now make the runfile executable and install CUDA but don't install the driver. Also the --override option helps to prevent some annoying errors, which could happen.

**IMPORTANT: Be sure to use the --no-opengl-libs option**
{% highlight bash %}
$ chmod +x cuda_8.0.44_linux-run
$ sudo sh cuda_8.0.44_linux-run --extract=~/Downloads/
$ sudo sh cuda_8.0.44_linux-run --override --no-opengl-libs
{% endhighlight %}
Now open your .bashrc
{% highlight bash %}
$ nano ~/.bashrc
{% endhighlight %}
and add the following lines:
{% highlight bash %}
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda
{% endhighlight %}

If the SCP operation is complete, extract it to the right locations.
{% highlight bash %}
$ sudo tar -xzvf cudnn-8.0-linux-x64-v5.1.tgz
$ sudo cp cuda/include/cudnn.h /usr/local/cuda/include
$ sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
$ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
{% endhighlight %}

Reboot one more time:

{% highlight bash %}
$ chmod +x ~/Downloads/Linux-x86_64/367.57/NVIDIA-Linux-x86_64-367.57.run
$ sudo sh ~/Linux-x86_64/367.57/NVIDIA-Linux-x86_64-367.57.run --no-opengl-files
$ sudo reboot
{% endhighlight %}

Now git clone Tensorflow and start the configuration:
{% highlight bash %}
$ git clone https://github.com/tensorflow/tensorflow
$ cd ~/tensorflow
$ ./configure
{% endhighlight %}

I used  **/usr/bin/python3.5** for the Python binary and **/usr/local/lib/python3.5/dist-packages** for the path, Cuda SDK **8.0**, cudnn **5.1.5**, compiled without cloud-support and OpenCL but with GPU support of course. Computing capabilities for the instance are are **3.0**.

Okay, now we compile everything:
{% highlight bash %}
$ bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
{% endhighlight %}
This will take some time. You could watch this video while you're waiting:
<iframe width="1120" height="630" src="https://www.youtube.com/embed/oQbei5JGiT8" frameborder="0" allowfullscreen></iframe>
Build the wheel:
{% highlight bash %}
$ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
{% endhighlight %}
If you want, you can use a virtual environment:
{% highlight bash %}
$ python3 -m venv --system-site-packages ~/tensorflow
$ source ~/tensorflow/bin/activate
$ pip3 install /tmp/tensorflow_pkg/tensorflow-0.11.0rc2-cp35-cp35m-linux_x86_64.whl
{% endhighlight %}
Installing OpenAI Gym is pretty straight forward, cause the people at OpenAI and all other contributers have done an amazing job (:
Sometimes there are some problems with Box2D. If you want to be sure, follow the instructions below.
We already installed Swig (we need 3.x before compiling Box2D).
Git clone Pybox2d
{% highlight bash %}
$ git clone https://github.com/pybox2d/pybox2d.git
{% endhighlight %}
Build and install it:
{% highlight bash %}
$ cd pybox2d
$ python setup.py build
$ python setup.py install
{% endhighlight %}
Installing OpenAI Gym should now be no trouble at all. We already installed most of the dependencies but I copy-pasted everything from their github instructions just to be sure:
{% highlight bash %}
$ git clone https://github.com/openai/gym.git
$ sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
$ pip install -e '.[all]'
{% endhighlight %}
If you had problems running OpenAI Gym headless using xvfb as X-server it should now work, if you do as explained by [Trevor Blackwell](https://github.com/tlbtlbtlb) in [this](https://github.com/openai/gym/issues/247#issuecomment-232731446) post (the GLX option is active by default):
{% highlight bash %}
$ xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python XXX.py
{% endhighlight %}
Have fun (:

If you run into any troubles, just let me know. I'm always happy to help.
