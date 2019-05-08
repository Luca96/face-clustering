# Face Clustering
Clusterize your faces according to similar features. FVAB final project [UL19].

## Project Structure

* **data/:** will contain the dataset?, test data? ...
* **weights/:** will contain the weights of the trained models.
* **src/:** all **future** python code.
  * `model.py`: model creation (architecture)
  * `train.py`: model training.
  * `test.py`: model evaluation, clustering evaluation, qualitative results.
  * `clustering.py`: clustering stuff.
  * `globals.py`: define shared (global) variables.
  * `main.py`: main script that glue everything together.
  * **utils/:** 
    * `cv.py`: OpenCV helpers (like crop, resize) - if needed.
    * `data.py`: loading dataset, splitting, data augmentation, etc.
    * `triplets.py`: triplet selection criteria for model loss function.
    * `loss.py`: model's custom loss function.

## Usage



## TODOs

- [ ] Custom loss function 
- [ ] Adding more TODOs..
- [ ] Dataset exploration
- [ ] Model for face-embeddings
- [ ] Fine-tuning model
- [ ] basic clustering
- [ ] Model evaluation
- [ ] Clustering evaluation
- [ ] Functional Dependencies discovery
- [ ] FD-driven clustering
- [ ] Evaluation of final code