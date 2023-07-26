# Exploring digital biomarkers of illness activity in mood episodes: hypotheses generating and model development study.

Code for paper "[Exploring digital biomarkers of illness activity in mood episodes: hypotheses generating and model development study](https://preprints.jmir.org/preprint/45405/)" published in JMIR mHealth and uHealth.

Authors: Gerard Anmella, Filippo Corponi*, Bryan M. Li*, Ariadna Mas, Miriam Sanabra, Isabella Pacchiarotti, Marc Valentí, Iria Grande, Antoni Benabarre,  Anna Giménez-Palomo, Marina Garriga, Isabel Agasi, Anna Bastidas, Myriam Cavero, Tabatha Fernández-Plaza, Nestor Arbelo, Miquel Bioque, Clemente García-Rizo, Norma Verdolini, Santiago Madero, Andrea Murrru, Silvia Amoretti, Anabel Martínez-Aran, Victoria Ruiz, Giovanna Fico, Michele De Prisco, Vincenzo Oliva,  Aleix Solanes,  Joaquim Radua,  Ludovic Samalin,  Allan Young, Eduard Vieta, Antonio Vergari, Diego Hidalgo-Mazzei. 
```
@article{anmella2023exploring,
  title={Exploring digital biomarkers of illness activity in mood episodes: hypotheses generating and model development study.},
  author={Anmella, Gerard and Corponi, Filippo and Li, Bryan M and Mas, Ariadna and Sanabra, Miriam and Pacchiarotti, Isabella and Valent{\'\i}, Marc and Grande, Iria and Benabarre, Antoni and Gim{\'e}nez-Palomo, Anna and others},
  journal={JMIR Mhealth and Uhealth},
  year={2023}
}
```

## Installation
- Create a new [conda](https://conda.io/en/latest/) environment with Python 3.8.
  ```bash
  conda create -n timebase python=3.8
  ```
- Activate `timebase` virtual environment
  ```bash
  conda activate timebase
  ```
- Install all dependencies and packages with `setup.sh` script, works on both Linus and macOS.
  ```bash
  sh setup.sh
  ```

## Data
- [dataset/README.md](dataset/README.md) details the structure of the dataset.

## Supervised Learning
### Aim 1: detection of the severity of an acute affective episode at the intra-individual level
- We herewith present the pipeline for Depression MDD (in-sample `MDD_is.yaml`, out-of sample `MDD_oos.yaml`). The same pipeline applies to Mania MD, Depression BD, Mixed features BD.

- The following command trains a model to classify severity states as specified in `MDD_is.yaml`:
  
  ```
  python classifier_train.py --dataset dataset/raw_data --output_dir runs/MDD_is --mode 0 --config configs/MDD_is.yaml --batch_size 8 --model bilstm --num_units 128 --epochs 150 --clear_output_dir --save_plots
  ```
- Run `tensorboard` to visualize training performance
  ```
  tensorboard --logdir runs/MDD_is
  ```
- For out-of-sample testing, launch the following command whichs uses trained model from above to classify severity states as specified in `MDD_oos.yaml`:
  ```
  python classifier_config.py --path2model runs/MDD_is --config configs/MDD_oos.yaml
  ```
- To perform dataset search on `MDD_is.yaml`, use the Python script `classifier_sesarch.py`
  ```
  python classifier_search.py --output_dir runs/MDD_is_search --config configs/MDD_is.yaml --model bilstm --save_plots
  ```


### Aim 2: identification of the polarity of an acute affective episode and euthymia among different individuals


- Assess the degree to which a deep learning model can distiguish between different healthy controls:
  ```
  python classifier_train.py --dataset dataset/raw_data --output_dir runs/HC --mode 0 --config configs/HC.yaml --batch_size 8 --model bilstm --num_units 256 --epochs 200 --clear_output_dir --save_plots
  ```

- Assess in- and out-of-sample performance at the between-individual level, using subjects with different mood disorder diagnoses. 
  
  In-sample:
  ```
  python classifier_train.py --dataset dataset/raw_data --output_dir runs/INTER_is --mode 0 --config configs/INTER_is.yaml --batch_size 8 --model bilstm --num_units 256 --epochs 200 --clear_output_dir --save_plots
  ```
  Out-of-sample:
  ```
  python classifier_config.py --path2model runs/INTER_is --config configs/INTER_oos.yaml
  ```



## Permutation Channel Importance
- The following command runs permutation channel importance:
  ```bash
  python classifier_permutation_feature_importance.py --experiment_dir runs/registration_study --algorithm transformer
  ```
It is assumed that all relevant experiments have been carried out so that the related results can be found under `runs/registration_study`:

  ```
  registration_study/
    - bi_dep/
      - 001_bilstm/
      - 002_transformer/
    - ...
    - uni_dep/
      - 001_bilstm/
      - 002_transformer/
  ```


No folder other than the experiments output should be placed in  `runs/registration_study`. The script outputs a plot and a dictionary with summary for channel importance.

 For a description of permuation feature importance please refer to [christophm](https://christophm.github.io/interpretable-ml-book/feature-importance.html). In our implementation, feature importance measures the percentage drop (positive value in terms of feature importance) or increase (negative value in terms of feature importance) in test set accuracy upon permutation of the feature under investigation. 

