conda create -n $[environment_name] python=3.5
conda activate $[environment_name]
conda install --channel https://conda.anaconda.org/menpo opencv3
conda install tensorflow matplotlib pillow keras -y 
conda install jupyter -y && pip install keras_tqdm