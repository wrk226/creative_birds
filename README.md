# 2D-to-3D-Evolution-Transfer
## Visualizations
- Evolutionary Birds from Different Angles
![rot](https://user-images.githubusercontent.com/111099396/185549797-c33fe6cb-2509-4811-b7a6-240ac9e0cc25.gif)
- Texture Evolution with Different Switch Gate
![tex](https://user-images.githubusercontent.com/111099396/185549807-85631823-3d8a-479a-a119-29551cc22602.gif)
- Shape Evolution with Different Scale Factor (Alpha)
![alpha](https://user-images.githubusercontent.com/111099396/185549805-0ed0d64b-1e96-4d4f-b0ba-d62b9f517261.gif)

## Prerequisites
- Download code & pre-trained model:
Git clone the code by:
```
git clone https://github.com/anonymous-226/2D-to-3D-Evolution-Transfer $ROOTPATH
```
The pretrained model can be found from [here](https://drive.google.com/file/d/1Agf_G9OaCvXPoenRK5vpj3VckuFPGRMg/view?usp=sharing), which should be unzipped in `$ROOTPATH`.
- Install packages:
```
conda create -n evo_trans python=3.6
conda activate evo_trans
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirements.txt
```

- Install external tools:

```
cd $ROOTPATH/2D-to-3D-Evolution-Transfer/dependency
unzip meshzoo-0.4.3.zip
cd meshzoo-0.4.3
python setup.py install
```

```
cd $ROOTPATH/2D-to-3D-Evolution-Transfer/dependency
unzip neural_renderer.zip
cd neural_renderer
python setup.py install
```

## Run the test
Run the following command from the `$ROOTPATH/2D-to-3D-Evolution-Transfer` directory:
```
python -m evo_trans.experiments.test_df2
```
The result can be found in `$ROOTPATH/2D-to-3D-Evolution-Transfer/evo_trans/cachedir/visualization` directory.
