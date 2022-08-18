# DRGNet

## Prerequisites
```
conda create -n evo_trans python=3.6
conda activate evo_trans
```
- Install packages:
```
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirements.txt
```

- Install external tools:

```
cd $ROOTPATH/dependency/meshzoo-0.4.3
python setup.py install
```

```
cd $ROOTPATH/dependency/neural_renderer
python setup.py install
```

## Run the test
python -m evo_trans.experiments.test_df2
