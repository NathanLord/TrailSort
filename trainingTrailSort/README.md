
# Set up virtual environment for the Jupyter notebook

```python -m venv venv```


```.\venv\Scripts\activate```

<br>

# Install Dependencies
```pip install -r requirements.txt```

or 
```
pip install tensorflow
pip install pillow
pip install matplotlib
pip install jupyter
pip install ipykernel
```
<br>

# Kernel for Jupyter Notebook after activating venv
```python -m ipykernel install --user --name=venv --display-name "Python (tensorflow_env)"```

## Check for Kernal
```jupyter kernelspec list```

