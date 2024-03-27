import my_fiftyone_script as fo
import fiftyone.zoo as foz
## load dataset
dataset = foz.load_zoo_dataset("open-images-v7")
session = fo.launch_app(dataset)