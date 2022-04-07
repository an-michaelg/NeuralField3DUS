# NeuralField3DUS
NeuralField3DUS - Learning Neural Field Representations for 3D Ultrasound

3D representation:
![3D representation](./supplementary_material/3dus_result_baseline.gif)

Meta-learning:
![Meta-learning](./supplementary_material/training_progression_us.png)

Required software packages:
- pytorch GPU
- numpy
- matplotlib
- PIL
- scipy
- wandb

To run, launch train_3dus.py or train_image_regression.py
Change path to datasets in dataloader.get_dataset_paths
Modify the config_dict to turn off meta-learning
